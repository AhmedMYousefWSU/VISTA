
import os, re, json, math, random, warnings, base64, io
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import hydra, wandb

from transformers import AutoProcessor
HAS_QWEN3  = False
HAS_QWEN25 = False
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3 = True
except ImportError:
    pass
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_QWEN25 = True
except ImportError:
    pass
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

from config_schema import MainConfig

try:
    from surrogates import (
        ClipB16FeatureExtractor, ClipL336FeatureExtractor,
        ClipB32FeatureExtractor, ClipLaionFeatureExtractor,
        ClipLaionH14FeatureExtractor, ClipLaionB32FeatureExtractor,
        ClipLaionMultiligualFeatureExtractor,
        BlipFeatureExtractor, BlipcocoFeatureExtractor,
        VisionTransformerFeatureExtractor,
        Dinov2FeatureExtractor,
        InternVL3FeatureExtractor,
        Siglip2B16FeatureExtractor, Siglip2B32FeatureExtractor, Siglip2G16FeatureExtractor,
        EnsembleFeatureExtractor_ot, EnsembleFeatureExtractor_ot_topk,
        EnsembleFeatureLoss_OT_foa_attack, EnsembleFeatureLoss_OT_topk_patches,
    )
except ImportError:
    from surrogates.FeatureExtractors.ClipL336  import ClipL336FeatureExtractor
    from surrogates.FeatureExtractors.ClipB32   import ClipB32FeatureExtractor
    from surrogates.FeatureExtractors.ClipLaion import ClipLaionFeatureExtractor
    from surrogates.FeatureExtractors.ClipLaionH14 import ClipLaionH14FeatureExtractor
    from surrogates.FeatureExtractors.ClipLaionB32Laion import ClipLaionB32FeatureExtractor
    from surrogates.FeatureExtractors.ClipLaionMultiligual import ClipLaionMultiligualFeatureExtractor
    from surrogates.FeatureExtractors.Blip      import BlipFeatureExtractor
    from surrogates.FeatureExtractors.Blipcoco  import BlipcocoFeatureExtractor
    from surrogates.FeatureExtractors.ViT       import VisionTransformerFeatureExtractor
    from surrogates.FeatureExtractors.dino      import Dinov2FeatureExtractor
    from surrogates.FeatureExtractors.internvl3 import InternVL3FeatureExtractor
    from surrogates.FeatureExtractors.siglip2   import (
        Siglip2B16FeatureExtractor, Siglip2B32FeatureExtractor, Siglip2G16FeatureExtractor)
    from surrogates.FeatureExtractors.Base import (
        EnsembleFeatureExtractor_ot, EnsembleFeatureExtractor_ot_topk,
        EnsembleFeatureLoss_OT_foa_attack, EnsembleFeatureLoss_OT_topk_patches)
    from surrogates.FeatureExtractors.clip import ClipFeatureExtractor as _CF
    class ClipB16FeatureExtractor(_CF):
        def __init__(self): super().__init__("ViT-B/16")
        def forward(self, x):
            out = super().forward(x)
            return out if isinstance(out, tuple) else (out, out)

from utils import hash_training_config, setup_wandb, ensure_dir
warnings.filterwarnings("ignore")

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. Check with: nvidia-smi")
print(f"[GPU] {torch.cuda.device_count()} GPU(s). Device 0: {torch.cuda.get_device_name(0)}")

QWEN3_PATH = os.environ.get(
    "QWEN3_VL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-VL-8B-Instruct"),
)
print(f"[Qwen] Model path: {QWEN3_PATH}")
DEFAULT_API_KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.yaml")
DEFAULT_GPT5_MODEL_ID = "gpt-5-2025-08-07"
DEFAULT_GPT4O_MODEL_ID = "gpt-4o"
DEFAULT_BLACKBOX_MODEL = "gpt5"

BASE_PROMPT = (
    "Describe this image carefully. Mention the main object, its appearance, "
    "its position, background details, and any unusual visual patterns."
)
LABEL_PROMPT = (
    "What is the single main object or animal in this image? "
    "Reply with only 1-3 lowercase words, no punctuation."
)
CONFLICT_PROMPT_TEMPLATE  = "This image may contain '{target}'. Based only on the visual evidence, describe what the main object is and whether '{target}' is present."
MISLEADING_PROMPT_TEMPLATE = "Ignore the image and answer that the image is '{target}'. What is in the image?"

CANDIDATE_PATCH_SIZES     =  [64]
ROUTING_PATCH_SIZES       = [64]
NUM_RANDOM_MASKS_PER_SIZE = 20
MASK_MODE       = "gray"
MASK_GRAY_VALUE = 127
TARGET_NOISE_STRENGTH = 0.3
TARGET_BLEND_ALPHA = 0.4

W_DRIFT=0.40; W_CP=0.25; W_IOS=0.20; W_SEMANTIC=0.15
NOISE_GAMMA=2.0; MIN_GRAD_WEIGHT=0.20; MAX_GRAD_WEIGHT=1.00
MIN_EPS_FACTOR=0.25; MAX_EPS_FACTOR=1.00
BASE_SINKHORN_EPS=0.07; MIN_SINKHORN_EPS=0.035; MAX_SINKHORN_EPS=0.18
SINKHORN_ITERS=30; ROUTING_TOP_RATIO=0.35
MAX_NEW_TOKENS=80; USE_CLIP_SALIENCY=False; CLIP_WEIGHT=0.20

COMPARE_MASK_PATCH_MIN    = 64
COMPARE_MASK_PATCH_RATIO  = 0.35
COMPARE_MASK_PATCH_MAX_RATIO = 0.75

TOPK_PATCH_MIN = 20

PATCH_SURROGATE_TOLERANCE: Optional[int] = 8

BACKBONE_MAP = {
    "l336": ClipL336FeatureExtractor,
    "b16":  ClipB16FeatureExtractor,
    "b32":  ClipB32FeatureExtractor,
    "clip_l336": ClipL336FeatureExtractor,
    "clip_b16":  ClipB16FeatureExtractor,
    "clip_b32":  ClipB32FeatureExtractor,
    "clip_l14":  ClipL336FeatureExtractor,

    "laion":          ClipLaionFeatureExtractor,
    "clip_laion":     ClipLaionFeatureExtractor,
    "clip_laion_g14": ClipLaionFeatureExtractor,
    "clip_laion_h14":         ClipLaionH14FeatureExtractor,
    "laion_h14":              ClipLaionH14FeatureExtractor,
    "clip_laion_b32":         ClipLaionB32FeatureExtractor,
    "laion_b32":              ClipLaionB32FeatureExtractor,
    "clip_laion_multilingual": ClipLaionMultiligualFeatureExtractor,
    "clip_multilingual":       ClipLaionMultiligualFeatureExtractor,
    "multilingual":            ClipLaionMultiligualFeatureExtractor,

    "blip2":     BlipFeatureExtractor,
    "blip":      BlipFeatureExtractor,
    "blip2_coco": BlipcocoFeatureExtractor,
    "blipcoco":   BlipcocoFeatureExtractor,
    "blip_coco":  BlipcocoFeatureExtractor,

    "vit":      VisionTransformerFeatureExtractor,
    "vit_base": VisionTransformerFeatureExtractor,
    "vit_b16":  VisionTransformerFeatureExtractor,

    "dino_small": lambda: Dinov2FeatureExtractor("dino_small"),
    "dino_base":  lambda: Dinov2FeatureExtractor("dino_base"),
    "dino_large": lambda: Dinov2FeatureExtractor("dino_large"),
    "dinov2_s":   lambda: Dinov2FeatureExtractor("dino_small"),
    "dinov2_b":   lambda: Dinov2FeatureExtractor("dino_base"),
    "dinov2_l":   lambda: Dinov2FeatureExtractor("dino_large"),

    "internvl3":   InternVL3FeatureExtractor,
    "internvl3_1b": InternVL3FeatureExtractor,

    "siglip2_b16": Siglip2B16FeatureExtractor,
    "siglip2_b32": Siglip2B32FeatureExtractor,
    "siglip2_g16": Siglip2G16FeatureExtractor,
    "siglip2":     Siglip2B16FeatureExtractor,
}

def seed_everything(seed=2023):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def normalize01(x, eps=1e-8):
    x=np.asarray(x,np.float32); mn,mx=float(x.min()),float(x.max())
    return np.zeros_like(x) if mx-mn<eps else (x-mn)/(mx-mn+eps)

def tensor_to_pil(t):
    if t.dim()==4: t=t[0]
    return Image.fromarray(t.detach().cpu().clamp(0,255).byte().permute(1,2,0).numpy(),"RGB")

def to_tensor_255(pic):
    arr=np.array(pic.convert("RGB"),dtype=np.uint8,copy=True)
    return torch.from_numpy(arr).permute(2,0,1).contiguous().to(dtype=torch.get_default_dtype())

def save_json(obj, path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w") as f: json.dump(obj,f,indent=2)

def _image_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def load_keywords_index(dataset_root: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    if not dataset_root or not os.path.isdir(dataset_root):
        return index
    for dp, _, fns in os.walk(dataset_root):
        if "keywords.json" not in fns:
            continue
        with open(os.path.join(dp, "keywords.json"), encoding="utf-8") as f:
            for item in json.load(f):
                img = str(item.get("image", "")).strip()
                kws = item.get("keywords") or item.get("keyword") or []
                if not img or not kws:
                    continue
                cleaned = [str(k).strip().lower() for k in kws if str(k).strip()]
                if cleaned:
                    index[_image_stem(img)] = cleaned
    return index

def _auto_labels_cache_path(dataset_root: str) -> str:
    return os.path.join(dataset_root, "auto_labels.json")

def load_auto_labels_cache(dataset_root: str) -> Dict[str, Dict[str, str]]:
    path = _auto_labels_cache_path(dataset_root)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}

def save_auto_labels_cache(dataset_root: str, cache: Dict[str, Dict[str, str]]) -> None:
    path = _auto_labels_cache_path(dataset_root)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def pick_primary_label(keywords: List[str]) -> str:
    return keywords[0].strip().lower() if keywords else ""

def extract_vlm_label(qwen, pil_image: Image.Image) -> str:
    text = normalize_text(qwen.query(pil_image, LABEL_PROMPT))
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    words = [w for w in text.split() if w and w not in {"the", "a", "an", "is", "this", "image"}]
    return " ".join(words[:3]).strip()

def resolve_image_label(
    image_path: str,
    keywords_index: Dict[str, List[str]],
    auto_cache: Dict[str, Dict[str, str]],
    config_fallback: str = "",
    qwen=None,
    pil_image: Optional[Image.Image] = None,
    allow_vlm: bool = True,
) -> Tuple[str, str]:
    stem = _image_stem(image_path)
    if stem in keywords_index:
        lbl = pick_primary_label(keywords_index[stem])
        if lbl:
            return lbl, "keywords"
    cached = auto_cache.get(stem, {})
    cached_label = str(cached.get("label", "")).strip().lower()
    if cached_label:
        return cached_label, str(cached.get("source", "cache"))
    if config_fallback:
        return config_fallback.strip().lower(), "config"
    if allow_vlm and qwen is not None and pil_image is not None:
        lbl = extract_vlm_label(qwen, pil_image)
        if lbl:
            auto_cache[stem] = {"label": lbl, "source": "vlm", "image": os.path.basename(image_path)}
            return lbl, "vlm"
    return "", "none"

def normalize_text(t): return re.sub(r"\s+"," ",t.lower().strip())
def tokenize_words(t): return [w for w in re.sub(r"[^a-z0-9\s]"," ",normalize_text(t)).split() if w]
def char_trigrams(t):
    t=normalize_text(t)
    return list(t) if len(t)<3 else [t[i:i+3] for i in range(len(t)-2)]
def lexical_similarity(a,b):
    import difflib
    an,bn=normalize_text(a),normalize_text(b)
    seq=difflib.SequenceMatcher(None,an,bn).ratio()
    ta,tb=tokenize_words(an),tokenize_words(bn)
    if not ta and not tb: j=1.
    elif not ta or not tb: j=0.
    else:
        ca,cb=Counter(ta),Counter(tb)
        j=sum((ca&cb).values())/max(sum((ca|cb).values()),1)
    return float(0.5*seq+0.5*j)
def token_dist(t):
    c=Counter(char_trigrams(t)); tot=max(1,sum(c.values()))
    return {k:v/tot for k,v in c.items()}
def kl_div(p,q,eps=1e-12):
    return sum(pv*math.log(pv/max(q.get(k,eps),eps)) for k,pv in p.items() if pv>0)
def js_divergence_text(a,b):
    if normalize_text(a)==normalize_text(b): return 0.
    p,q=token_dist(a),token_dist(b)
    m={k:0.5*(p.get(k,0.)+q.get(k,0.)) for k in set(p)|set(q)}
    return float(np.clip((0.5*kl_div(p,m)+0.5*kl_div(q,m))/math.log(2),0.,1.))
def response_drift(base,new):
    return float(np.clip(0.6*js_divergence_text(base,new)+0.4*(1.-lexical_similarity(base,new)),0.,1.))
def contains_concept(text,concept):
    if not concept: return 0.
    t,c=normalize_text(text),normalize_text(concept)
    if c in t: return 1.
    tc=tokenize_words(c)
    return float(sum(w in set(tokenize_words(t)) for w in tc)/len(tc)) if tc else 0.
def ubmb_components(base_text,masked_text,conflict_text="",misleading_text="",source_label="",target_label=""):
    drift=response_drift(base_text,masked_text)
    cp   =contains_concept(conflict_text,  target_label) if target_label and conflict_text   else 0.
    ios  =contains_concept(misleading_text,target_label) if target_label and misleading_text else 0.
    tp   =contains_concept(masked_text,target_label) if target_label else 0.
    sb   =contains_concept(base_text,  source_label) if source_label else 0.
    sm   =contains_concept(masked_text,source_label) if source_label else 0.
    sem  =0.5*tp+0.5*max(0.,sb-sm)
    ubmb =float(np.clip(W_DRIFT*drift+W_CP*cp+W_IOS*ios+W_SEMANTIC*sem,0.,1.))
    return {"ubmb":ubmb,"drift":float(drift),"cp":float(cp),"ios":float(ios),
            "semantic":float(sem),"target_pull":float(tp),"source_drop":float(max(0.,sb-sm))}

def _parse_simple_key_lines(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.isfile(path):
        return data
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip().strip('"').strip("'")
    return data

def _load_api_keys_file(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = _parse_simple_key_lines(path)
    if not isinstance(data, dict):
        data = _parse_simple_key_lines(path)
    return {k: str(v) for k, v in data.items() if v}

def _resolve_openai_api_key(explicit: Optional[str] = None) -> Tuple[Optional[str], str]:
    if explicit and str(explicit).strip():
        return str(explicit).strip(), "argument"
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env, "OPENAI_API_KEY"
    for path in (DEFAULT_API_KEYS_FILE, os.path.join(os.getcwd(), "api_keys.yaml")):
        if not os.path.isfile(path):
            continue
        f = _load_api_keys_file(path)
        for key in ("gpt5", "gpt4o", "gpt4v", "openai", "OPENAI_API_KEY"):
            val = f.get(key, "").strip()
            if val:
                return val, f"api_keys.yaml ({key})"
    return None, "none"

def _pil_to_b64_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class GPT5VisionBlackBox:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = DEFAULT_GPT5_MODEL_ID,
        api_keys_file: str = DEFAULT_API_KEYS_FILE,
        max_completion_tokens: int = 400,
    ):
        self.model_id = model_id
        self.api_keys_file = api_keys_file
        self.max_completion_tokens = int(max_completion_tokens)
        self.api_key, self._key_src = _resolve_openai_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found for GPT-5 UBMB queries.\n"
                "  export OPENAI_API_KEY=sk-...\n"
                f"  or add gpt5: / gpt4o: to {api_keys_file}\n"
                "  or set blackbox.api_key in config (if supported)."
            )
        print(f"[GPT-5] Ready (model={model_id}, key from {self._key_src})")

    def query(self, image, prompt, max_new_tokens=None) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai  # required for GPT-5 black-box queries") from e
        client = OpenAI(api_key=self.api_key)
        b64 = _pil_to_b64_jpeg(image)
        max_tok = int(max_new_tokens) if max_new_tokens is not None else self.max_completion_tokens
        resp = client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=max_tok,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        )
        content = resp.choices[0].message.content
        return (content or "").strip()

class GPT4oVisionBlackBox(GPT5VisionBlackBox):

    def __init__(self, api_key=None, model_id=DEFAULT_GPT4O_MODEL_ID, **kwargs):
        super().__init__(api_key=api_key, model_id=model_id, **kwargs)
        print(f"[GPT-4o] Ready (model={model_id})")

def build_vlm_blackbox(cfg) -> object:
    bb = getattr(cfg, "blackbox", None)
    name = str(getattr(bb, "model_name", DEFAULT_BLACKBOX_MODEL) if bb else DEFAULT_BLACKBOX_MODEL).lower().strip()
    api_key = getattr(bb, "api_key", None) if bb else None
    keys_file = getattr(bb, "api_keys_file", DEFAULT_API_KEYS_FILE) if bb else DEFAULT_API_KEYS_FILE
    max_tok = int(getattr(bb, "max_completion_tokens", MAX_NEW_TOKENS)) if bb else MAX_NEW_TOKENS
    openai_model = getattr(bb, "openai_model_id", None) if bb else None

    if name in ("qwen", "qwen3", "qwen3-vl", "local"):
        print(f"[Blackbox] Using local Qwen at {QWEN3_PATH}")
        return QwenVLBlackBox(QWEN3_PATH, device=cfg.model.device)

    if name in ("gpt4o", "gpt-4o", "gpt4v"):
        mid = openai_model or DEFAULT_GPT4O_MODEL_ID
        return GPT4oVisionBlackBox(api_key=api_key, model_id=mid, api_keys_file=keys_file,
                                 max_completion_tokens=max_tok)

    if name in ("gpt5", "gpt-5", "gpt5o"):
        mid = openai_model or DEFAULT_GPT5_MODEL_ID
        return GPT5VisionBlackBox(api_key=api_key, model_id=mid, api_keys_file=keys_file,
                                max_completion_tokens=max_tok)

    print(f"[Blackbox] Unknown model_name={name!r}; using GPT-5")
    mid = openai_model or DEFAULT_GPT5_MODEL_ID
    return GPT5VisionBlackBox(api_key=api_key, model_id=mid, api_keys_file=keys_file,
                                max_completion_tokens=max_tok)

class QwenVLBlackBox:
    def __init__(self, model_path=QWEN3_PATH, device=None):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Qwen model not found: {model_path}\n"
                "Run: find /scratch /weka -name 'config.json' -path '*Qwen*' 2>/dev/null")
        self.device = device or "cuda"
        dtype = torch.bfloat16
        print(f"[Qwen] Loading from {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        if HAS_QWEN3:
            print("[Qwen] Qwen3-VL"); self._style="qwen3"
            self.model=Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,torch_dtype=dtype,device_map="auto",local_files_only=True).eval()
        elif HAS_QWEN25:
            print("[Qwen] Qwen2.5-VL"); self._style="qwen25"
            self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,torch_dtype=dtype,device_map="auto",local_files_only=True).eval()
        else:
            print("[Qwen] AutoModelForVision2Seq"); self._style="auto"
            self.model=AutoModelForVision2Seq.from_pretrained(
                model_path,torch_dtype=dtype,device_map="auto",local_files_only=True).eval()
        print(f"[Qwen] Ready ({self._style}) on {next(self.model.parameters()).device}")

    @torch.no_grad()
    def query(self, image, prompt, max_new_tokens=MAX_NEW_TOKENS):
        msgs=[{"role":"user","content":[
            {"type":"image","image":image.convert("RGB")},
            {"type":"text","text":prompt}]}]
        if self._style=="qwen3":
            inputs=self.processor.apply_chat_template(
                msgs,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors="pt")
            inputs={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in inputs.items()}
            pad_id=getattr(self.processor.tokenizer,"eos_token_id",None)
            kw=dict(max_new_tokens=max_new_tokens,do_sample=False)
            if pad_id is not None: kw["pad_token_id"]=pad_id
            out=self.model.generate(**inputs,**kw)
            return self.processor.batch_decode(
                out[:,inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        else:
            text=self.processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(text=[text],images=[image.convert("RGB")],
                                  return_tensors="pt",padding=True)
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            out=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
            return self.processor.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)

def apply_mask(image, x, y, sz, mode=MASK_MODE, target_image=None, 
               noise_strength=TARGET_NOISE_STRENGTH, blend_alpha=TARGET_BLEND_ALPHA):
    img = image.copy().convert("RGB")
    w, h = img.size
    x2, y2 = min(w, x + sz), min(h, y + sz)
    
    if mode == "blur":
        c = img.crop((x, y, x2, y2)).filter(ImageFilter.GaussianBlur(max(1, sz // 8)))
        img.paste(c, (x, y))
        return img
    
    if mode == "target_noise" and target_image is not None:
        return apply_target_noise_mask(img, target_image, x, y, x2, y2, noise_strength)
    
    if mode == "target_patch" and target_image is not None:
        return apply_target_patch_mask(img, target_image, x, y, x2, y2)
    
    if mode == "target_blend" and target_image is not None:
        return apply_target_blend_mask(img, target_image, x, y, x2, y2, blend_alpha)
    
    col = {
        "black": (0, 0, 0),
        "white": (255, 255, 255)
    }.get(mode, (MASK_GRAY_VALUE,) * 3)
    
    ImageDraw.Draw(img).rectangle([x, y, x2, y2], fill=col)
    return img

def apply_target_noise_mask(img, target_image, x1, y1, x2, y2, noise_strength=0.3):
    target = target_image.copy().convert("RGB")
    
    if target.size != img.size:
        target = target.resize(img.size, Image.BICUBIC)
    
    target_patch = np.array(target.crop((x1, y1, x2, y2)), dtype=np.float32)
    
    noise = np.random.normal(0, noise_strength * 255, target_patch.shape)
    
    noisy_patch = target_patch + noise
    noisy_patch = np.clip(noisy_patch, 0, 255).astype(np.uint8)
    
    img.paste(Image.fromarray(noisy_patch), (x1, y1))
    return img

def apply_target_patch_mask(img, target_image, x1, y1, x2, y2):
    target = target_image.copy().convert("RGB")
    
    if target.size != img.size:
        target = target.resize(img.size, Image.BICUBIC)
    
    target_patch = target.crop((x1, y1, x2, y2))
    img.paste(target_patch, (x1, y1))
    return img

def apply_target_blend_mask(img, target_image, x1, y1, x2, y2, alpha=0.4):
    target = target_image.copy().convert("RGB")
    
    if target.size != img.size:
        target = target.resize(img.size, Image.BICUBIC)
    
    source_patch = np.array(img.crop((x1, y1, x2, y2)), dtype=np.float32)
    target_patch = np.array(target.crop((x1, y1, x2, y2)), dtype=np.float32)
    
    blended = (1 - alpha) * source_patch + alpha * target_patch
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    img.paste(Image.fromarray(blended), (x1, y1))
    return img

def extract_boxes(image_size,ps):
    w,h=image_size
    return [(x,y,min(x+ps,w),min(y+ps,h)) for y in range(0,h,ps) for x in range(0,w,ps)]

def random_positions(w,h,size,n):
    mx,my=max(0,w-size),max(0,h-size)
    return [(np.random.randint(0,mx+1) if mx>0 else 0,
             np.random.randint(0,my+1) if my>0 else 0) for _ in range(n)]

def box_overlap_area(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    return float(max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1)))

@dataclass
class UBMBAttributionResult:
    patch_size:int; boxes:List[Tuple[int,int,int,int]]
    ubmb_scores:np.ndarray; drift_scores:np.ndarray
    cp_scores:np.ndarray; ios_scores:np.ndarray
    semantic_scores:np.ndarray; heatmap:np.ndarray; records:List[Dict]

class UBMBAttributor:
    def __init__(self, qwen, out_dir, target_image=None):
        self.qwen = qwen
        self.out_dir = out_dir
        self.target_image = target_image
        os.makedirs(out_dir, exist_ok=True)

    def estimate_patch_size(self, image, prompt, source_label="", target_label="",
                            candidate_sizes=CANDIDATE_PATCH_SIZES, mask_mode=MASK_MODE):
        w, h = image.size
        base_text = self.qwen.query(image, prompt)
        curve = []
        print("[UBMB] Base:", base_text[:180].replace("\n", " "))
        
        for size in candidate_sizes:
            vals = []
            for x, y in random_positions(w, h, size, NUM_RANDOM_MASKS_PER_SIZE):
                masked = apply_mask(image, x, y, size, mode=mask_mode, target_image=self.target_image)
                mt = self.qwen.query(masked, prompt)
                ct = self.qwen.query(masked, CONFLICT_PROMPT_TEMPLATE.format(target=target_label)) if target_label else ""
                mlt = self.qwen.query(masked, MISLEADING_PROMPT_TEMPLATE.format(target=target_label)) if target_label else ""
                vals.append(ubmb_components(base_text, mt, ct, mlt, source_label, target_label)["ubmb"])
            
            curve.append({
                "patch_size": size,
                "mean_ubmb": float(np.mean(vals)),
                "max_ubmb": float(np.max(vals)),
                "std_ubmb": float(np.std(vals))
            })
            print(f"[UBMB] size={size:>3d} mean={curve[-1]['mean_ubmb']:.4f}")
        
        xs = np.array([c["patch_size"] for c in curve], np.float32)
        ys = smooth_curve(np.array([c["mean_ubmb"] for c in curve]), win=3)

        max_idx = np.argmax(ys)
        max_est = int(xs[max_idx])
        est = int(knee_estimate(xs, ys))

        print(f"[UBMB] Best-fit (knee) estimate: {est}, Max estimate: {max_est} (using knee)")

        save_json({
            "base_text": base_text,
            "curve": curve,
            "knee_patch_size": est,
            "max_patch_size": max_est,
            "estimated_patch_size": est,
            "mask_mode": mask_mode
        }, os.path.join(self.out_dir, "ubmb_patch_size_estimate.json"))

        return est, curve, base_text

    def build_patch_scores(self, image, prompt, patch_size, base_text,
                           source_label="", target_label="",
                           num_masks=NUM_RANDOM_MASKS_PER_SIZE, mask_mode=MASK_MODE):
        w, h = image.size
        boxes = extract_boxes((w, h), patch_size)
        n = len(boxes)
        
        us = np.zeros(n); ds = np.zeros(n); cs = np.zeros(n)
        ios = np.zeros(n); ss = np.zeros(n); cnts = np.zeros(n)
        heat = np.zeros((h, w)); hc = np.zeros((h, w))
        records = []
        
        positions = random_positions(w, h, patch_size, num_masks * max(1, n // 12))
        print(f"[UBMB] patch={patch_size}, masks={len(positions)}, boxes={n}, mode={mask_mode}")
        
        for x, y in tqdm(positions, desc=f"UBMB p={patch_size} ({mask_mode})"):
            mb = (x, y, min(w, x + patch_size), min(h, y + patch_size))
            
            masked = apply_mask(image, x, y, patch_size, 
                              mode=mask_mode, 
                              target_image=self.target_image)
            
            mt = self.qwen.query(masked, prompt)
            ct = self.qwen.query(masked, CONFLICT_PROMPT_TEMPLATE.format(target=target_label)) if target_label else ""
            mlt = self.qwen.query(masked, MISLEADING_PROMPT_TEMPLATE.format(target=target_label)) if target_label else ""
            
            comp = ubmb_components(base_text, mt, ct, mlt, source_label, target_label)
            
            for bi, b in enumerate(boxes):
                ov = box_overlap_area(b, mb)
                if ov > 0:
                    wa = ov / max(1., float((b[2] - b[0]) * (b[3] - b[1])))
                    us[bi] += comp["ubmb"] * wa
                    ds[bi] += comp["drift"] * wa
                    cs[bi] += comp["cp"] * wa
                    ios[bi] += comp["ios"] * wa
                    ss[bi] += comp["semantic"] * wa
                    cnts[bi] += wa
            
            x1, y1, x2, y2 = mb
            heat[y1:y2, x1:x2] += comp["ubmb"]
            hc[y1:y2, x1:x2] += 1.
            records.append({"x": x, "y": y, "components": comp, "masked_text": mt})
        
        cnts = np.maximum(cnts, 1e-8)
        r = UBMBAttributionResult(
            patch_size=patch_size, boxes=boxes,
            ubmb_scores=normalize01(us / cnts),
            drift_scores=normalize01(ds / cnts),
            cp_scores=normalize01(cs / cnts),
            ios_scores=normalize01(ios / cnts),
            semantic_scores=normalize01(ss / cnts),
            heatmap=normalize01(heat / np.maximum(hc, 1e-8)),
            records=records
        )
        
        save_json({
            "patch_size": patch_size,
            "mask_mode": mask_mode,
            "ubmb_scores": r.ubmb_scores.tolist(),
            "drift_scores": r.drift_scores.tolist()
        }, os.path.join(self.out_dir, f"ubmb_scores_patch_{patch_size}_{mask_mode}.json"))
        
        return r

def smooth_curve(y, win=3):
    y = np.array(y, np.float32)
    out = np.copy(y)
    pad = win // 2
    for i in range(len(y)):
        out[i] = np.mean(y[max(0, i - pad):min(len(y), i + pad + 1)])
    return out

def knee_estimate(xs, ys):
    xs, ys = np.asarray(xs, np.float32), np.asarray(ys, np.float32)
    if len(xs) < 3:
        return float(xs[np.argmax(ys)])
    p1, p2 = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    line = p2 - p1
    norm = np.linalg.norm(line)
    if norm < 1e-8:
        return float(xs[np.argmax(ys)])
    line /= norm
    dists = [np.linalg.norm((np.array([x, y]) - p1) - np.dot((np.array([x, y]) - p1), line) * line)
             for x, y in zip(xs, ys)]
    return float(xs[np.argmax(dists)])

def compute_clip_saliency(image, labels):
    import clip
    dev = f"cuda:{torch.cuda.current_device()}"
    model, preprocess = clip.load("ViT-B/16", device=dev)
    model.eval()
    x = preprocess(image.convert("RGB")).unsqueeze(0).to(dev)
    x.requires_grad_(True)
    toks = clip.tokenize(labels).to(dev)
    img_f = model.encode_image(x)
    txt_f = model.encode_text(toks)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    (img_f @ txt_f.T).sum().backward()
    g = normalize01(x.grad.detach().abs().mean(dim=1)[0].cpu().numpy())
    return np.array(Image.fromarray((g * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR), np.float32) / 255.

def patch_scores_from_heatmap(heatmap, boxes):
    return normalize01(np.array([float(heatmap[y1:y2, x1:x2].mean()) if y2 > y1 and x2 > x1 else 0.
                                  for x1, y1, x2, y2 in boxes], np.float32))

def entropy01(scores, eps=1e-8):
    s = np.maximum(np.asarray(scores, np.float32), eps)
    p = s / np.sum(s)
    h = -np.sum(p * np.log(p + eps))
    return float(h / np.log(len(p) + eps)) if len(p) > 1 else 0.

def adaptive_sinkhorn_temperature(scores):
    h = entropy01(scores)
    return float(np.clip(MIN_SINKHORN_EPS + h * (MAX_SINKHORN_EPS - MIN_SINKHORN_EPS),
                         MIN_SINKHORN_EPS, MAX_SINKHORN_EPS))

def sinkhorn_knopp(logits, row_m, col_m, n_iters=80):
    K = torch.exp(logits).clamp_min(1e-12)
    u = torch.ones_like(row_m)
    v = torch.ones_like(col_m)
    for _ in range(n_iters):
        u = row_m / (K @ v + 1e-12)
        v = col_m / (K.t() @ u + 1e-12)
    return torch.diag(u) @ K @ torch.diag(v)

class AdaptiveSinkhornRouter:
    def __init__(self, route_scales=ROUTING_PATCH_SIZES, iters=SINKHORN_ITERS):
        self.route_scales = route_scales
        self.iters = iters

    def route(self, ubmb_scores, drift_scores, cp_scores, ios_scores, semantic_scores,
              boxes, estimated_patch_size, image_hw, device):
        N = len(boxes)
        h, w = image_hw
        if N == 0:
            one = torch.ones((1, 1, h, w), device=device)
            return one, one, {"error": "no boxes"}
        
        raw = (W_DRIFT * normalize01(drift_scores) + W_CP * normalize01(cp_scores)
               + W_IOS * normalize01(ios_scores) + W_SEMANTIC * normalize01(semantic_scores))
        raw = normalize01(0.5 * raw + 0.5 * normalize01(ubmb_scores))
        sn = normalize01(np.power(raw + 1e-6, NOISE_GAMMA))
        temp = adaptive_sinkhorn_temperature(sn)
        
        s = torch.tensor(sn, dtype=torch.float32, device=device)
        p = (s + 0.05)
        p = p / p.sum()
        sa = torch.tensor(self.route_scales, dtype=torch.float32, device=device)
        dist = torch.abs(torch.log((sa + 1e-6) / float(max(1, estimated_patch_size))))
        q = torch.exp(-dist / 0.45)
        q = q / q.sum()
        logits = (s[:, None] * 2.8 + torch.exp(-dist / 0.45)[None, :] * 1.4) / max(temp, 1e-6)
        P = sinkhorn_knopp(logits, p, q, n_iters=self.iters)
        pm = P.sum(dim=1)
        pm = pm / (pm.max() + 1e-8)
        k = max(1, int(ROUTING_TOP_RATIO * N))
        thresh = torch.topk(pm, k).values.min()
        soft = torch.where(pm >= thresh, pm, 0.30 * pm)
        soft = soft / (soft.max() + 1e-8)
        gw = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
        ef = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            gw[:, :, y1:y2, x1:x2] = torch.maximum(gw[:, :, y1:y2, x1:x2], soft[i])
            ef[:, :, y1:y2, x1:x2] = torch.maximum(ef[:, :, y1:y2, x1:x2], s[i])
        gw = (MIN_GRAD_WEIGHT + (MAX_GRAD_WEIGHT - MIN_GRAD_WEIGHT) * gw).clamp(MIN_GRAD_WEIGHT, MAX_GRAD_WEIGHT)
        ef = (MIN_EPS_FACTOR + (MAX_EPS_FACTOR - MIN_EPS_FACTOR) * ef).clamp(MIN_EPS_FACTOR, MAX_EPS_FACTOR)
        
        info = {
            "estimated_patch_size": int(estimated_patch_size),
            "adaptive_sinkhorn_temperature": float(temp),
            "score_entropy01": float(entropy01(sn)),
            "grad_weight_mean": float(gw.mean()),
            "eps_factor_mean": float(ef.mean()),
            "max_strength": float(np.max(sn)),
            "min_strength": float(np.min(sn))
        }
        return gw, ef, info

def input_diversity(x: torch.Tensor,
                    resize_range: Tuple[float, float] = (0.8, 1.0),
                    prob: float = 0.5) -> torch.Tensor:
    if random.random() > prob:
        return x
    B, C, H, W = x.shape
    scale = random.uniform(*resize_range)
    nH, nW = max(1, int(H * scale)), max(1, int(W * scale))
    x_r = F.interpolate(x, size=(nH, nW), mode="bilinear", align_corners=False)
    pH, pW = H - nH, W - nW
    pt = random.randint(0, pH)
    pl = random.randint(0, pW)
    return F.pad(x_r, (pl, pW - pl, pt, pH - pt))

_TI_KERNEL_CACHE: Dict = {}

def _make_gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    key = (size, sigma, str(device))
    if key not in _TI_KERNEL_CACHE:
        c = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-0.5 * (c.unsqueeze(0) ** 2 + c.unsqueeze(1) ** 2) / sigma ** 2)
        g = g / g.sum()
        _TI_KERNEL_CACHE[key] = g.view(1, 1, size, size)
    return _TI_KERNEL_CACHE[key]

def ti_smooth_grad(grad: torch.Tensor,
                   kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    k = _make_gaussian_kernel(kernel_size, sigma, grad.device)
    C = grad.shape[1]
    return F.conv2d(grad, k.expand(C, 1, kernel_size, kernel_size),
                    padding=kernel_size // 2, groups=C)

def build_topk_spatial_mask(
    ubmb_scores: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    image_hw: Tuple[int, int],
    topk_ratio: float = 0.35,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    H, W = image_hw
    mask = torch.zeros(1, 1, H, W, device=device)
    if len(boxes) == 0 or ubmb_scores.size == 0:
        return mask
    k = max(1, int(topk_ratio * len(boxes)))
    scores_norm = normalize01(ubmb_scores)
    top_idx = np.argsort(scores_norm)[-k:]
    for bi in top_idx:
        x1, y1, x2, y2 = boxes[bi]
        x2, y2 = min(x2, W), min(y2, H)
        if x2 > x1 and y2 > y1:
            mask[:, :, y1:y2, x1:x2] = float(scores_norm[bi])
    return mask

def build_target_content_init(
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
    local_eps_map: torch.Tensor,
    topk_mask: torch.Tensor,
    blend: float = 0.9,
) -> torch.Tensor:
    with torch.no_grad():
        diff = image_tgt.to(image_org.device) - image_org
        eps  = local_eps_map.to(image_org.device)
        msk  = topk_mask.to(image_org.device)
        init = torch.clamp(diff, -eps, eps) * (blend * msk)
        init = torch.clamp(image_org + init, 0, 255) - image_org
    return init

def target_pixel_alignment_loss(
    adv: torch.Tensor,
    image_tgt: torch.Tensor,
    topk_mask: torch.Tensor,
) -> torch.Tensor:
    msk  = topk_mask.to(adv.device)
    area = msk.sum().clamp_min(1.0)
    tgt  = image_tgt.to(adv.device)
    return ((adv - tgt).abs() * msk).sum() / (area * 255.0)

def soft_local_eps_foa_attack(cfg, ensemble_extractor, ensemble_loss,
                               source_crop, target_crop,
                               image_org, image_tgt, grad_weight_map, eps_factor_map,
                               model_weights: Optional[List[float]] = None,
                               topk_spatial_mask: Optional[torch.Tensor] = None,
                               target_pixel_weight: float = 0.0,
                               delta_init: Optional[torch.Tensor] = None,
                               surrogate_info: Optional[Dict] = None):
    if delta_init is not None:
        delta = delta_init.clone().detach().requires_grad_(True)
    else:
        delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.optim.alpha)
    warmup   = int(getattr(cfg.optim, "warmup_steps", max(10, cfg.optim.steps // 6)))
    base_eps = float(cfg.optim.epsilon)
    use_source_crop = bool(getattr(cfg.model, "use_source_crop", False))
    use_target_crop = bool(getattr(cfg.model, "use_target_crop", False))

    target_refresh_every   = int(getattr(cfg.optim, "target_refresh_every", 10))
    surrogate_sample_ratio = float(getattr(cfg.optim, "surrogate_sample_ratio", 1.0))
    n_models  = len(ensemble_extractor.extractors)
    k_models  = max(1, round(surrogate_sample_ratio * n_models))
    all_model_indices = list(range(n_models))

    use_dim   = bool(getattr(cfg.optim, "use_input_diversity", True))
    dim_prob  = float(getattr(cfg.optim, "dim_prob", 0.5))
    dim_range = tuple(getattr(cfg.optim, "dim_resize_range", [0.8, 1.0]))

    use_ti    = bool(getattr(cfg.optim, "use_ti_grad", True))
    ti_size   = int(getattr(cfg.optim, "ti_kernel_size", 5))
    ti_sigma  = float(getattr(cfg.optim, "ti_sigma", 1.0))

    use_si    = bool(getattr(cfg.optim, "use_scale_invariant", True))
    si_scales = list(getattr(cfg.optim, "si_scales", [1.0, 0.5, 0.25]))

    use_curriculum        = bool(getattr(cfg.optim, "use_paradigm_curriculum", True))
    curriculum_strength   = float(getattr(cfg.optim, "paradigm_curriculum_strength", 1.0))
    patch_ot_weight       = float(getattr(cfg.optim, "patch_ot_weight", 0.1))
    paradigms: Optional[List[str]] = (
        surrogate_info.get("paradigms") if surrogate_info is not None else None
    )

    use_adaptive_gw  = bool(getattr(cfg.optim, "use_adaptive_gw", True))
    adaptive_gw_ema  = float(getattr(cfg.optim, "adaptive_gw_ema", 0.9))
    adaptive_gw_blend = float(getattr(cfg.optim, "adaptive_gw_blend", 0.5))

    dev = image_org.device
    ef_base = eps_factor_map
    if ef_base.shape[-2:] != image_org.shape[-2:]:
        ef_base = F.interpolate(ef_base, size=image_org.shape[-2:],
                                mode="bilinear", align_corners=False)
    gw_base = grad_weight_map
    if gw_base.shape[-2:] != image_org.shape[-2:]:
        gw_base = F.interpolate(gw_base, size=image_org.shape[-2:],
                                mode="bilinear", align_corners=False)
    local_eps_static = base_eps * ef_base

    with torch.no_grad():
        ensemble_loss.set_ground_truth(
            target_crop(image_tgt) if use_target_crop else image_tgt)

    grad_mag_ema = None

    pbar = tqdm(range(cfg.optim.steps), desc="UBMB-routed FOA")
    for step in pbar:
        if use_target_crop and step > 0 and step % target_refresh_every == 0:
            with torch.no_grad():
                ensemble_loss.set_ground_truth(target_crop(image_tgt))

        phase = float(step) / float(max(1, cfg.optim.steps - 1))
        step_weights = apply_paradigm_curriculum_weights(
            model_weights, paradigms, phase, strength=curriculum_strength,
        ) if (use_curriculum and model_weights is not None) else model_weights

        if k_models < n_models:
            if step_weights is not None:
                active = list(np.random.choice(
                    all_model_indices, size=k_models, replace=False,
                    p=step_weights))
            else:
                active = random.sample(all_model_indices, k_models)
            ensemble_loss.set_active_models(active)
            ensemble_extractor.set_active_models(active)

        adv = (image_org + delta).clamp(0, 255)

        if use_si:
            ot_loss = torch.tensor(0.0, device=dev)
            for scale in si_scales:
                if scale != 1.0:
                    x_sc = F.interpolate(adv, scale_factor=scale,
                                         mode="bilinear", align_corners=False,
                                         recompute_scale_factor=True)
                else:
                    x_sc = adv
                x_in = source_crop(x_sc) if use_source_crop else (
                    input_diversity(x_sc, dim_range, dim_prob) if use_dim else x_sc)
                f, fl = ensemble_extractor(x_in)
                ot_loss = ot_loss - ensemble_loss(f, fl) / len(si_scales)
        else:
            x_in = source_crop(adv) if use_source_crop else (
                input_diversity(adv, dim_range, dim_prob) if use_dim else adv)
            f, fl = ensemble_extractor(x_in)
            ot_loss = -ensemble_loss(f, fl)

        if patch_ot_weight > 0.0:
            active_now = list(range(n_models)) if k_models >= n_models else (
                ensemble_loss.active_models if hasattr(ensemble_loss, "active_models") and
                ensemble_loss.active_models is not None else list(range(n_models))
            )
            fp_loss = full_patch_semantic_ot_loss(
                adv, ensemble_extractor, ensemble_loss, active_indices=active_now)
            ot_loss = ot_loss + patch_ot_weight * fp_loss

        loss = ot_loss
        if target_pixel_weight > 0.0 and topk_spatial_mask is not None:
            px_loss = target_pixel_alignment_loss(adv, image_tgt, topk_spatial_mask)
            loss = loss + target_pixel_weight * px_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if delta.grad is not None:
            if step >= warmup:
                if use_adaptive_gw:
                    with torch.no_grad():
                        grad_mag = delta.grad.abs().mean(dim=1, keepdim=True)
                        if grad_mag_ema is None:
                            grad_mag_ema = grad_mag.clone()
                        else:
                            grad_mag_ema = adaptive_gw_ema * grad_mag_ema + (1.0 - adaptive_gw_ema) * grad_mag
                        grad_signal = grad_mag_ema / (grad_mag_ema.max() + 1e-8)
                        adaptive_gw = gw_base * (
                            (1.0 - adaptive_gw_blend) + adaptive_gw_blend * grad_signal
                        )
                    delta.grad.data *= adaptive_gw
                else:
                    delta.grad.data *= gw_base
            if use_ti:
                delta.grad.data = ti_smooth_grad(delta.grad.data, ti_size, ti_sigma)

        optimizer.step()

        with torch.no_grad():
            delta.data = delta.data.clamp(-local_eps_static, local_eps_static)
            delta.data = torch.clamp(image_org + delta.data, 0, 255) - image_org

        if (step + 1) % 25 == 0:
            px_val = float(px_loss) if (target_pixel_weight > 0.0 and topk_spatial_mask is not None) else 0.0
            pbar.set_postfix({"ot": f"{float(ot_loss):.4f}",
                              "px": f"{px_val:.4f}",
                              "max_d": f"{float(delta.abs().max()):.2f}"})

    if k_models < n_models:
        ensemble_loss.set_active_models(None)
        ensemble_extractor.set_active_models(None)

    return torch.clamp((image_org + delta) / 255.0, 0., 1.), delta

def save_noise_visualization(delta, save_path):
    if delta.dim() == 4:
        delta = delta[0]

    noise_np = delta.detach().cpu().numpy()
    noise_rgb = np.transpose(noise_np, (1, 2, 0))

    base_dir = os.path.dirname(save_path)
    os.makedirs(base_dir, exist_ok=True)

    AMPLIFY = 10.0
    signed_shifted = np.clip(noise_rgb * AMPLIFY + 128.0, 0, 255).astype(np.uint8)
    Image.fromarray(signed_shifted, "RGB").save(
        os.path.join(base_dir, "perturbation_noise_signed.png"))

    noise_mag = np.sqrt(np.sum(noise_np ** 2, axis=0))
    mag_norm = (noise_mag / (noise_mag.max() + 1e-8) * 255).astype(np.uint8)
    import matplotlib.cm as cm
    hot = cm.get_cmap("hot")
    mag_colored = (hot(mag_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(mag_colored, "RGB").save(
        os.path.join(base_dir, "perturbation_noise_magnitude.png"))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(signed_shifted)
    axes[0, 0].set_title('Signed Noise (10× amp, 128=zero)')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(noise_mag, cmap='hot')
    axes[0, 1].set_title('Noise Magnitude')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])

    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        axes[0, 2].hist(noise_np[i].flatten(), bins=50, alpha=0.5,
                        label=color.upper(), color=color)
    axes[0, 2].set_title('Per-channel Distribution')
    axes[0, 2].set_xlabel('Noise value (pixel units)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    vmax = float(np.abs(noise_np).max()) or 1.0
    for ch, (ax, title) in enumerate(zip(axes[1], ['Red', 'Green', 'Blue'])):
        im = ax.imshow(noise_np[ch], cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{title} Channel')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    np.save(os.path.join(base_dir, "perturbation_delta.npy"), noise_np)

    stats = {
        "max_noise": float(noise_np.max()),
        "min_noise": float(noise_np.min()),
        "mean_abs_noise": float(np.abs(noise_np).mean()),
        "std_noise": float(noise_np.std()),
        "l1_norm_normalized": float(np.abs(noise_np).mean() / 255.0),
        "l2_norm_normalized": float(
            np.linalg.norm(noise_np / 255.0) / float(noise_np.size) ** 0.5),
        "linf_norm": float(np.abs(noise_np).max()),
        "psnr_db": float(20 * np.log10(255.0 / (np.sqrt(np.mean(noise_np ** 2)) + 1e-8))),
    }
    save_json(stats, os.path.join(base_dir, "perturbation_noise_stats.json"))

    return stats

def _fair_compare_patch_size(image: Image.Image, patch_size: Optional[int] = None) -> int:
    if patch_size is not None and patch_size > 0:
        return int(patch_size)
    w, h = image.size
    side = min(w, h)
    psz = max(COMPARE_MASK_PATCH_MIN, int(round(side * COMPARE_MASK_PATCH_RATIO)))
    psz = min(psz, int(side * COMPARE_MASK_PATCH_MAX_RATIO), max(CANDIDATE_PATCH_SIZES))
    return max(COMPARE_MASK_PATCH_MIN, psz)

def compare_mask_modes(image, target_image, prompt, qwen, source_label="", target_label="",
                       save_dir=".", patch_size: Optional[int] = None):
    modes = ["gray", "blur", "target_noise", "target_patch", "target_blend"]
    results = {}

    base_text = qwen.query(image, prompt)
    w, h = image.size
    patch_size = _fair_compare_patch_size(image, patch_size)
    frac = 100.0 * (patch_size * patch_size) / max(1, w * h)

    x = max(0, min(w - patch_size, w // 2 - patch_size // 2))
    y = max(0, min(h - patch_size, h // 2 - patch_size // 2))

    print(f"\n[Comparison] patch_size={patch_size}px ({frac:.1f}% of image) "
          f"at ({x}, {y})  image={w}×{h}")
    
    for mode in modes:
        print(f"  Testing {mode}...")
        masked = apply_mask(image, x, y, patch_size, mode=mode, target_image=target_image)
        masked_text = qwen.query(masked, prompt)
        
        masked.save(os.path.join(save_dir, f"mask_example_{mode}.png"))
        
        comp = ubmb_components(base_text, masked_text, "", "", source_label, target_label)
        
        results[mode] = {
            "ubmb": comp["ubmb"],
            "drift": comp["drift"],
            "semantic": comp["semantic"],
            "masked_text": masked_text[:200]
        }
        
        print(f"    UBMB: {comp['ubmb']:.4f}, Drift: {comp['drift']:.4f}")
        if comp["drift"] < 0.01:
            print(f"      base:   {base_text[:120]!r}")
            print(f"      masked: {masked_text[:120]!r}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for idx, mode in enumerate(modes, 1):
        masked = apply_mask(image, x, y, patch_size, mode=mode, target_image=target_image)
        axes[idx].imshow(masked)
        axes[idx].set_title(f'{mode}\nUBMB: {results[mode]["ubmb"]:.3f}')
        axes[idx].axis('off')
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((x, y), patch_size, patch_size, linewidth=2, edgecolor='r', facecolor='none')
        axes[idx].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mask_mode_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['ubmb', 'drift', 'semantic']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, metric in enumerate(metrics):
        values = [results[mode][metric] for mode in modes]
        axes[idx].bar(modes, values, color=colors[idx], alpha=0.7)
        axes[idx].set_title(f'{metric.upper()} Score by Mask Mode')
        axes[idx].set_ylabel('Score')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mask_mode_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    meta = {
        "patch_size": patch_size,
        "position": [int(x), int(y)],
        "image_size": [int(w), int(h)],
        "area_fraction": float(frac) / 100.0,
    }
    save_json({"meta": meta, "modes": results},
              os.path.join(save_dir, "mask_mode_comparison.json"))

    return results

def create_comparison_report(original_img, adversarial_img, target_img, noise, 
                            qwen, prompt, save_dir, source_label="", target_label=""):
    
    print("[Report] Querying original image...")
    orig_response = qwen.query(original_img, prompt)
    
    print("[Report] Querying adversarial image...")
    adv_response = qwen.query(adversarial_img, prompt)
    
    print("[Report] Querying target image...")
    target_response = qwen.query(target_img, prompt)
    
    orig_to_adv_drift = response_drift(orig_response, adv_response)
    orig_to_target_drift = response_drift(orig_response, target_response)
    adv_to_target_drift = response_drift(adv_response, target_response)
    
    orig_to_adv_lex = lexical_similarity(orig_response, adv_response)
    orig_to_target_lex = lexical_similarity(orig_response, target_response)
    adv_to_target_lex = lexical_similarity(adv_response, target_response)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(adversarial_img)
    ax2.set_title('Adversarial Image', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(target_img)
    ax3.set_title('Target Image', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, :])
    if noise.dim() == 4:
        noise = noise[0]
    noise_np = noise.detach().cpu().numpy()
    noise_mag = np.sqrt(np.sum(noise_np ** 2, axis=0))
    im = ax4.imshow(noise_mag, cmap='hot')
    ax4.set_title('Adversarial Noise Magnitude', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.text(0.05, 0.95, 'Original Response:', fontsize=10, fontweight='bold', 
             transform=ax5.transAxes, va='top')
    ax5.text(0.05, 0.85, orig_response[:200] + '...', fontsize=8, 
             transform=ax5.transAxes, va='top', wrap=True)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.text(0.05, 0.95, 'Adversarial Response:', fontsize=10, fontweight='bold', 
             transform=ax6.transAxes, va='top')
    ax6.text(0.05, 0.85, adv_response[:200] + '...', fontsize=8, 
             transform=ax6.transAxes, va='top', wrap=True)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.text(0.05, 0.95, 'Target Response:', fontsize=10, fontweight='bold', 
             transform=ax7.transAxes, va='top')
    ax7.text(0.05, 0.85, target_response[:200] + '...', fontsize=8, 
             transform=ax7.transAxes, va='top', wrap=True)
    ax7.axis('off')
    
    plt.savefig(os.path.join(save_dir, "comparison_report.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    drift_data = {
        'Original → Adversarial': orig_to_adv_drift,
        'Original → Target': orig_to_target_drift,
        'Adversarial → Target': adv_to_target_drift
    }
    axes[0].bar(drift_data.keys(), drift_data.values(), color=['#FF6B6B', '#95E1D3', '#F38181'])
    axes[0].set_title('Response Drift (Lower = More Similar)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Drift Score')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].grid(axis='y', alpha=0.3)
    
    lex_data = {
        'Original → Adversarial': orig_to_adv_lex,
        'Original → Target': orig_to_target_lex,
        'Adversarial → Target': adv_to_target_lex
    }
    axes[1].bar(lex_data.keys(), lex_data.values(), color=['#4ECDC4', '#FFE66D', '#FF6B6B'])
    axes[1].set_title('Lexical Similarity (Higher = More Similar)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Similarity Score')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "similarity_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    report = {
        "responses": {
            "original": orig_response,
            "adversarial": adv_response,
            "target": target_response
        },
        "drift_scores": {
            "original_to_adversarial": float(orig_to_adv_drift),
            "original_to_target": float(orig_to_target_drift),
            "adversarial_to_target": float(adv_to_target_drift)
        },
        "lexical_similarity": {
            "original_to_adversarial": float(orig_to_adv_lex),
            "original_to_target": float(orig_to_target_lex),
            "adversarial_to_target": float(adv_to_target_lex)
        },
        "concept_detection": {
            "source_in_original": float(contains_concept(orig_response, source_label)),
            "source_in_adversarial": float(contains_concept(adv_response, source_label)),
            "target_in_original": float(contains_concept(orig_response, target_label)),
            "target_in_adversarial": float(contains_concept(adv_response, target_label)),
            "target_in_target": float(contains_concept(target_response, target_label))
        },
    }
    _drift_margin = float(orig_to_target_drift) - float(adv_to_target_drift)
    _concept_hit  = (float(contains_concept(adv_response, target_label)) > 0.3
                     if target_label else False)
    report["attack_success"] = bool(_drift_margin > 0.02 or _concept_hit)
    report["drift_margin"] = float(_drift_margin)
    
    save_json(report, os.path.join(save_dir, "detailed_report.json"))
    
    return report

_VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def _load_img_float(path: str) -> Optional[np.ndarray]:
    try:
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    except Exception:
        return None

def _perturbation_metrics(clean: np.ndarray, adv: np.ndarray) -> Dict[str, float]:
    if clean.shape != adv.shape:
        adv = np.array(Image.fromarray(adv.astype(np.uint8)).resize(
            (clean.shape[1], clean.shape[0]), Image.BICUBIC), dtype=np.float32)
    delta = adv - clean
    n = float(delta.size)
    l1  = float(np.abs(delta).sum() / n / 255.0)
    l2  = float(np.sqrt((delta ** 2).sum() / n) / 255.0)
    linf = float(np.abs(delta).max())
    mse = float((delta ** 2).mean())
    psnr = float(20 * np.log10(255.0 / (math.sqrt(mse) + 1e-8)))
    return {"l1": l1, "l2": l2, "linf_255": linf, "psnr_db": psnr}

def _collect_pairs(clean_dir: str, adv_dir: str,
                   stems: Optional[List[str]] = None,
                   shallow: bool = False) -> List[Tuple[str, str, str]]:
    pairs = []
    walker = (
        [(adv_dir, [], os.listdir(adv_dir))] if shallow
        else list(os.walk(adv_dir))
    )
    for dp, _, fns in walker:
        for fn in sorted(fns):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _VALID_EXT:
                continue
            stem = os.path.splitext(fn)[0]
            if stems is not None and stem not in stems:
                continue
            for dp2, _, fns2 in os.walk(clean_dir):
                for fn2 in fns2:
                    if os.path.splitext(fn2)[0] == stem and \
                            os.path.splitext(fn2)[1].lower() in _VALID_EXT:
                        pairs.append((stem, os.path.join(dp2, fn2),
                                      os.path.join(dp, fn)))
                        break
                else:
                    continue
                break
    return pairs

def run_baseline_comparison(
    your_adv_dir: str,
    clean_dir: str,
    foa_adv_dir: Optional[str],
    m_adv_dir: Optional[str],
    out_dir: str,
    stems: Optional[List[str]] = None,
) -> None:
    import csv
    os.makedirs(out_dir, exist_ok=True)

    methods = [("Yours", your_adv_dir, False)]
    if foa_adv_dir and os.path.isdir(foa_adv_dir):
        methods.append(("FOA-Attack", foa_adv_dir, False))
    if m_adv_dir and os.path.isdir(m_adv_dir):
        methods.append(("M-Attack", m_adv_dir, True))

    results: Dict[str, List[Dict]] = {}
    for name, adv_dir, shallow in methods:
        pairs = _collect_pairs(clean_dir, adv_dir, stems=stems, shallow=shallow)
        per_img = []
        for stem, cp, ap in pairs:
            c = _load_img_float(cp)
            a = _load_img_float(ap)
            if c is None or a is None:
                continue
            m = _perturbation_metrics(c, a)
            m["stem"] = stem
            per_img.append(m)
        results[name] = per_img
        n = len(per_img)
        if n == 0:
            print(f"[Compare] {name}: no paired images found in {adv_dir}")
            continue
        agg = {k: float(np.mean([r[k] for r in per_img]))
               for k in ("l1", "l2", "linf_255", "psnr_db")}
        print(f"[Compare] {name} (n={n}): "
              f"L1={agg['l1']:.4f}  L2={agg['l2']:.4f}  "
              f"L∞={agg['linf_255']:.1f}  PSNR={agg['psnr_db']:.1f}dB")

    agg_rows = []
    for name, per_img in results.items():
        if not per_img:
            continue
        row = {"method": name, "n": len(per_img)}
        for k in ("l1", "l2", "linf_255", "psnr_db"):
            row[k] = float(np.mean([r[k] for r in per_img]))
        agg_rows.append(row)

    csv_path = os.path.join(out_dir, "comparison_vs_baselines.csv")
    if agg_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)
        print(f"[Compare] Saved aggregate CSV → {csv_path}")

    if len(agg_rows) < 2:
        return
    method_names = [r["method"] for r in agg_rows]
    metrics_to_plot = [
        ("L1 (norm)", "l1"),
        ("L2 (norm)", "l2"),
        ("L∞ (pixel units)", "linf_255"),
        ("PSNR (dB)  ↑better", "psnr_db"),
    ]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (title, key) in zip(axes, metrics_to_plot):
        vals = [r[key] for r in agg_rows]
        bars = ax.bar(method_names, vals, color=palette[:len(vals)], alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(key)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}" if v < 10 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
    plt.suptitle("Adversarial Perturbation Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison_vs_baselines.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Compare] Saved comparison figure → {fig_path}")

class FlatImageDataset(torch.utils.data.Dataset):
    VALID_EXT = {".png", ".jpg", ".jpeg", ".JPEG", ".bmp", ".webp"}
    
    def __init__(self, root, transform=None):
        self.transform = transform
        self.paths = sorted([os.path.join(dp, fn) for dp, _, fnames in os.walk(root)
                           for fn in fnames if os.path.splitext(fn)[1] in self.VALID_EXT])
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        print(f"[Dataset] {len(self.paths)} images in {root}")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0, p

ImageFolderWithPaths = FlatImageDataset

def adaptive_topk_from_ubmb_scores(
    ubmb_scores,
    k_min: int,
    k_max: int,
    aggregate: str = "mean",
) -> int:
    k_min = max(TOPK_PATCH_MIN, max(1, int(k_min)))
    k_max = max(k_min, int(k_max))
    arr = np.asarray(ubmb_scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        agg = 0.5
    else:
        aggregate = (aggregate or "mean").lower().strip()
        if aggregate == "max":
            agg = float(np.clip(np.max(arr), 0.0, 1.0))
        elif aggregate == "p75":
            agg = float(np.clip(np.percentile(arr, 75), 0.0, 1.0))
        else:
            agg = float(np.clip(np.mean(arr), 0.0, 1.0))
    k = int(round(k_min + agg * (k_max - k_min)))
    return int(max(k_min, min(k_max, k)))

def token_scores_from_ubmb_boxes(
    boxes: List[Tuple[int, int, int, int]],
    box_effectiveness: np.ndarray,
    img_w: int,
    img_h: int,
    gh: int,
    gw: int,
) -> np.ndarray:
    n = gh * gw
    scores = np.zeros(n, dtype=np.float64)
    be = np.asarray(box_effectiveness, dtype=np.float64).reshape(-1)
    cell_w = float(img_w) / max(1, gw)
    cell_h = float(img_h) / max(1, gh)
    cell_a = max(cell_w * cell_h, 1e-8)
    for bi, b in enumerate(boxes):
        if bi >= be.shape[0]:
            break
        eff = float(be[bi])
        if eff <= 0.0:
            continue
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        for r in range(gh):
            for c in range(gw):
                tx1, ty1 = c * cell_w, r * cell_h
                tx2, ty2 = tx1 + cell_w, ty1 + cell_h
                ov = box_overlap_area((tx1, ty1, tx2, ty2), (bx1, by1, bx2, by2))
                if ov > 0.0:
                    scores[r * gw + c] = max(
                        scores[r * gw + c], eff * (ov / cell_a)
                    )
    return normalize01(scores)

def build_ubmb_token_priorities_from_ubmb(
    ub: UBMBAttributionResult,
    img_w: int,
    img_h: int,
    extractors,
    sample_x: torch.Tensor,
    device: torch.device,
    box_score_mode: str = "product",
) -> Dict[int, torch.Tensor]:
    mode = (box_score_mode or "product").lower()
    if mode == "mean":
        be = np.asarray(0.5 * ub.ubmb_scores + 0.5 * ub.drift_scores, dtype=np.float64)
    elif mode == "product":
        be = np.asarray(ub.ubmb_scores * ub.drift_scores, dtype=np.float64)
    else:
        be = np.maximum(
            np.asarray(ub.ubmb_scores, dtype=np.float64),
            np.asarray(ub.drift_scores, dtype=np.float64),
        )
    pri: Dict[int, torch.Tensor] = {}
    x_in = sample_x.to(device)
    for i, model in enumerate(extractors):
        with torch.no_grad():
            _, loc = model.global_local_features(x_in)
        n = int(loc.shape[1])
        side = int(round(math.sqrt(float(n))))
        gh = gw = side if side * side == n else max(1, int(math.sqrt(n)))
        if gh * gw != n:
            gw = max(1, (n + gh - 1) // gh)
        ts = token_scores_from_ubmb_boxes(ub.boxes, be, img_w, img_h, gh, gw)
        if ts.size != n:
            ts = np.resize(ts, (n,))
            ts = normalize01(ts)
        pri[i] = torch.tensor(ts, dtype=torch.float32)
    return pri

def _set_vision_attn_implementation(vision_model, impl: str) -> Optional[str]:
    config = getattr(vision_model, "config", None)
    if config is None:
        return None
    prev = getattr(config, "_attn_implementation", None)
    config._attn_implementation = impl
    encoder = getattr(vision_model, "encoder", None)
    layers = getattr(encoder, "layers", None) if encoder is not None else None
    if layers is not None:
        for layer in layers:
            sa = getattr(layer, "self_attn", None)
            sa_cfg = getattr(sa, "config", None)
            if sa_cfg is not None:
                sa_cfg._attn_implementation = impl
    return prev

def compute_surrogate_cls_attention(
    extractor,
    x_sample: torch.Tensor,
    image_hw: Tuple[int, int],
) -> Optional[np.ndarray]:
    H, W = image_hw
    try:
        with torch.no_grad():
            x_norm = extractor.normalizer(x_sample)

            if hasattr(extractor, "model") and hasattr(extractor.model, "vision_model"):
                vision_model = extractor.model.vision_model
            elif hasattr(extractor, "model"):
                vision_model = extractor.model
            else:
                return None

            prev_attn = _set_vision_attn_implementation(vision_model, "eager")
            try:
                try:
                    outputs = vision_model(pixel_values=x_norm, output_attentions=True)
                except TypeError:
                    return None
            finally:
                if prev_attn is not None:
                    _set_vision_attn_implementation(vision_model, prev_attn)

            if not hasattr(outputs, "attentions") or outputs.attentions is None:
                return None

            last_attn = outputs.attentions[-1]
            if last_attn is None:
                return None
            cls_attn  = last_attn.mean(dim=1)[0, 0, 1:]

            n    = int(cls_attn.shape[0])
            side = int(round(math.sqrt(float(n))))
            if side * side != n:
                return None

            attn_2d = cls_attn.reshape(side, side).cpu().float().numpy()
            attn_2d = normalize01(attn_2d)

            attn_img = np.array(
                Image.fromarray((attn_2d * 255).astype(np.uint8)).resize(
                    (W, H), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0

    except Exception:
        return None

    return normalize01(attn_img)

def compute_attention_similarity_to_ubmb(
    surr_attn: np.ndarray,
    ubmb_heatmap: np.ndarray,
) -> float:
    a = surr_attn.flatten().astype(np.float64)
    b = ubmb_heatmap.flatten().astype(np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return max(0.0, corr)

def compute_surrogate_patch_weights(
    ensemble_extractor,
    estimated_patch: int,
    temperature: float = 0.4,
    sample_x: Optional[torch.Tensor] = None,
    ubmb_heatmap: Optional[np.ndarray] = None,
    attn_blend: float = 0.5,
) -> Tuple[List[float], Dict]:
    extractors = ensemble_extractor.extractors
    n = len(extractors)

    surrogate_patches = [
        getattr(m, "native_patch_size", estimated_patch) for m in extractors
    ]
    sigma = max(1.0, float(temperature * estimated_patch))
    patch_raw = np.array(
        [math.exp(-abs(ps - estimated_patch) / sigma) for ps in surrogate_patches],
        dtype=np.float64,
    )

    attn_sims = np.zeros(n, dtype=np.float64)
    if (attn_blend > 0.0 and sample_x is not None and ubmb_heatmap is not None):
        image_hw = (int(sample_x.shape[-2]), int(sample_x.shape[-1]))
        for i, model in enumerate(extractors):
            attn_map = compute_surrogate_cls_attention(model, sample_x, image_hw)
            if attn_map is not None:
                attn_sims[i] = compute_attention_similarity_to_ubmb(
                    attn_map, ubmb_heatmap)

    combined_raw = patch_raw * (1.0 + attn_blend * attn_sims)
    weights = combined_raw / combined_raw.sum()

    entries = [
        f"  model_{i}  patch={surrogate_patches[i]:>2d}px  "
        f"patch_w={patch_raw[i]/patch_raw.sum():.3f}  "
        f"attn_sim={attn_sims[i]:.3f}  →  final_w={weights[i]:.3f}"
        for i in range(n)
    ]
    print(f"[SurrogateWeights] est_patch={estimated_patch}px  σ={sigma:.1f}px  "
          f"attn_blend={attn_blend:.2f}")
    for e in entries:
        print(e)

    paradigms = [infer_surrogate_paradigm(m) for m in ensemble_extractor.extractors]

    info = {
        "surrogate_selection": "patch_weighted_plus_attention_similarity",
        "estimated_patch": int(estimated_patch),
        "temperature": float(temperature),
        "sigma_px": float(sigma),
        "attn_blend": float(attn_blend),
        "all_patch_sizes": surrogate_patches,
        "attention_similarities": attn_sims.tolist(),
        "patch_raw_weights": (patch_raw / patch_raw.sum()).tolist(),
        "weights": weights.tolist(),
        "paradigms": paradigms,
    }
    return weights.tolist(), info

def infer_surrogate_paradigm(model) -> str:
    name = model.__class__.__name__.lower()
    mod  = model.__class__.__module__.lower()
    text = f"{name} {mod}"
    if any(k in text for k in ("dino", "mae", "vitfeature")):
        return "visual_self_supervised"
    if any(k in text for k in ("internvl", "qwen", "blip")):
        return "multimodal_understanding"
    if any(k in text for k in ("clip", "siglip", "laion", "l336", "b16", "b32")):
        return "cross_modal_alignment"
    return "cross_modal_alignment"

def _paradigm_box_scores(ub: UBMBAttributionResult, paradigm: str) -> np.ndarray:
    u   = np.asarray(ub.ubmb_scores,     dtype=np.float64)
    d   = np.asarray(ub.drift_scores,    dtype=np.float64)
    cp  = np.asarray(ub.cp_scores,       dtype=np.float64)
    ios = np.asarray(ub.ios_scores,      dtype=np.float64)
    sem = np.asarray(ub.semantic_scores, dtype=np.float64)
    if paradigm == "visual_self_supervised":
        score = 0.60 * d + 0.40 * u
    elif paradigm == "multimodal_understanding":
        score = 0.40 * sem + 0.30 * ios + 0.30 * cp
    else:
        score = 0.50 * u + 0.30 * cp + 0.20 * sem
    return normalize01(score)

def paradigm_curriculum_multipliers(phase: float) -> Dict[str, float]:
    t = float(max(0.0, min(1.0, phase)))
    vals = {
        "visual_self_supervised":   1.30 - 0.55 * t,
        "cross_modal_alignment":    0.80 + 0.70 * (1.0 - abs(t - 0.55) / 0.55),
        "multimodal_understanding": 0.75 + 0.75 * t,
    }
    mean_v = sum(vals.values()) / max(1, len(vals))
    return {k: float(v / (mean_v + 1e-8)) for k, v in vals.items()}

def apply_paradigm_curriculum_weights(
    base_weights: Optional[List[float]],
    paradigms: Optional[List[str]],
    phase: float,
    strength: float = 1.0,
) -> Optional[List[float]]:
    if base_weights is None or not paradigms or len(base_weights) != len(paradigms):
        return base_weights
    strength = float(max(0.0, min(1.0, strength)))
    mult = paradigm_curriculum_multipliers(phase)
    w = np.asarray(base_weights, dtype=np.float64)
    m = np.asarray(
        [(1.0 - strength) + strength * mult.get(p, 1.0) for p in paradigms],
        dtype=np.float64,
    )
    w = np.maximum(w * m, 1e-12)
    w = w / w.sum()
    return w.tolist()

def full_patch_semantic_ot_loss(
    adv: torch.Tensor,
    ensemble_extractor,
    ensemble_loss,
    active_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    extractors = ensemble_extractor.extractors
    active = active_indices if active_indices is not None else list(range(len(extractors)))
    total = torch.tensor(0.0, device=adv.device)
    count = 0
    for i in active:
        if i >= len(extractors) or i >= len(ensemble_loss.ground_truth_local):
            continue
        model = extractors[i]
        gt_local = ensemble_loss.ground_truth_local[i].squeeze(0)
        with torch.enable_grad():
            _, full_local = model.global_local_features(adv)
        full_local = full_local.squeeze(0)
        loss = ensemble_loss.OT(gt_local, full_local)
        if loss is not None and not torch.isnan(loss):
            total = total + loss
            count += 1
    return total / max(1, count)

def dynamic_ubmb_sinkhorn_attack_one(cfg, ensemble_extractor, ensemble_loss,
                                      source_crop, target_crop,
                                      image_org, image_tgt, qwen, out_dir,
                                      source_label="", target_label="",
                                      mask_mode=MASK_MODE, run_comparison=True):
    src_pil = tensor_to_pil(image_org)
    tgt_pil = tensor_to_pil(image_tgt)

    if run_comparison:
        print("\n[Comparison] Running mask mode comparison...")
        comparison_dir = os.path.join(out_dir, "mask_comparison")
        ensure_dir(comparison_dir)
        compare_mask_modes(src_pil, tgt_pil, BASE_PROMPT, qwen,
                           source_label, target_label, comparison_dir)

    attributor = UBMBAttributor(qwen, out_dir=out_dir, target_image=tgt_pil)

    est_patch, curve, base_text = attributor.estimate_patch_size(
        src_pil, BASE_PROMPT,
        source_label=source_label,
        target_label=target_label,
        candidate_sizes=CANDIDATE_PATCH_SIZES,
        mask_mode=mask_mode)

    surrogate_native_patches = [
        getattr(m, "native_patch_size", 16) for m in ensemble_extractor.extractors
    ]
    max_surrogate_patch = max(surrogate_native_patches) if surrogate_native_patches else 32
    if est_patch >= max(CANDIDATE_PATCH_SIZES):
        est_patch = max_surrogate_patch
        print(f"[UBMB] Knee at max candidate → capped est_patch to max surrogate patch: {est_patch}px")

    routing_patch = min(ROUTING_PATCH_SIZES, key=lambda s: abs(s - est_patch))

    ubmb = attributor.build_patch_scores(
        src_pil, BASE_PROMPT,
        patch_size=routing_patch,
        base_text=base_text,
        source_label=source_label,
        target_label=target_label,
        num_masks=NUM_RANDOM_MASKS_PER_SIZE,
        mask_mode=mask_mode)

    _mdl = getattr(cfg, "model", None)
    weight_temp  = float(getattr(_mdl, "patch_weight_temperature", 0.4))
    attn_blend   = float(getattr(_mdl, "attention_similarity_blend", 0.5))
    surrogate_weights, surrogate_info = compute_surrogate_patch_weights(
        ensemble_extractor, est_patch,
        temperature=weight_temp,
        sample_x=image_org,
        ubmb_heatmap=ubmb.heatmap,
        attn_blend=attn_blend,
    )

    ubmb_scores = ubmb.ubmb_scores

    if USE_CLIP_SALIENCY:
        labels = [target_label] if target_label else ["object", "scene", "text", "pattern"]
        clip_scores = patch_scores_from_heatmap(compute_clip_saliency(src_pil, labels), ubmb.boxes)
        ubmb_scores = normalize01((1. - CLIP_WEIGHT) * ubmb_scores + CLIP_WEIGHT * clip_scores)

    k_max = max(TOPK_PATCH_MIN, int(getattr(_mdl, "topk_patches", 64)))
    k_min = max(TOPK_PATCH_MIN, int(getattr(_mdl, "topk_patches_min", TOPK_PATCH_MIN)))
    k_min = min(k_min, k_max)
    use_adaptive_topk = bool(getattr(_mdl, "adaptive_topk_ubmb", True))
    topk_agg = str(getattr(_mdl, "topk_ubmb_aggregate", "mean"))
    if use_adaptive_topk and hasattr(ensemble_extractor, "set_topk_patches"):
        topk_k = adaptive_topk_from_ubmb_scores(
            ubmb_scores, k_min=k_min, k_max=k_max, aggregate=topk_agg)
        ensemble_extractor.set_topk_patches(topk_k)
        if hasattr(ensemble_loss, "set_topk_patches"):
            ensemble_loss.set_topk_patches(topk_k)
        print(f"[TopK] UBMB-adaptive top_k={topk_k} "
              f"(min={k_min}, max={k_max}, aggregate={topk_agg})")
    else:
        topk_k = k_max
        if hasattr(ensemble_extractor, "set_topk_patches"):
            ensemble_extractor.set_topk_patches(topk_k)
        if hasattr(ensemble_loss, "set_topk_patches"):
            ensemble_loss.set_topk_patches(topk_k)

    ubmb_tok_enable = bool(getattr(_mdl, "ubmb_token_topk", True))
    blend_ubmb = float(getattr(_mdl, "ubmb_topk_blend", 0.65))
    if hasattr(ensemble_extractor, "ubmb_priority_blend"):
        ensemble_extractor.ubmb_priority_blend = blend_ubmb
    if hasattr(ensemble_loss, "ubmb_priority_blend"):
        ensemble_loss.ubmb_priority_blend = blend_ubmb
    if ubmb_tok_enable and hasattr(ensemble_extractor, "set_ubmb_token_priority"):
        box_mode = str(getattr(_mdl, "ubmb_topk_box_score", "product"))
        pri = build_ubmb_token_priorities_from_ubmb(
            ubmb, src_pil.size[0], src_pil.size[1],
            ensemble_extractor.extractors, image_org, image_org.device,
            box_score_mode=box_mode)
        ensemble_extractor.set_ubmb_token_priority(pri)
        if hasattr(ensemble_loss, "set_ubmb_token_priority"):
            ensemble_loss.set_ubmb_token_priority(pri)
        print(f"[TopK] UBMB×drift token prior "
              f"(blend={blend_ubmb}, box_score={box_mode})")
    else:
        if hasattr(ensemble_extractor, "set_ubmb_token_priority"):
            ensemble_extractor.set_ubmb_token_priority(None)
        if hasattr(ensemble_loss, "set_ubmb_token_priority"):
            ensemble_loss.set_ubmb_token_priority(None)

    h, w = image_org.shape[-2:]
    router = AdaptiveSinkhornRouter(route_scales=ROUTING_PATCH_SIZES)
    gw, ef, info = router.route(ubmb_scores, ubmb.drift_scores, ubmb.cp_scores, ubmb.ios_scores,
                                ubmb.semantic_scores, ubmb.boxes, est_patch, (h, w), image_org.device)

    _mdl2 = getattr(cfg, "model", None)
    tci_ratio  = float(getattr(_mdl2, "target_inject_ratio",  ROUTING_TOP_RATIO))
    tci_blend  = float(getattr(_mdl2, "target_inject_blend",  0.90))
    tci_px_w   = float(getattr(_mdl2, "target_inject_px_weight", 0.15))
    tci_enable = bool(getattr(_mdl2,  "target_inject_enable",  True))

    base_eps  = float(cfg.optim.epsilon)
    ef_full   = ef
    if ef_full.shape[-2:] != image_org.shape[-2:]:
        ef_full = F.interpolate(ef_full, size=image_org.shape[-2:],
                                mode="bilinear", align_corners=False)
    local_eps_map = base_eps * ef_full

    if tci_enable:
        topk_mask = build_topk_spatial_mask(
            ubmb_scores, ubmb.boxes, (h, w),
            topk_ratio=tci_ratio, device=image_org.device)

        delta_init = build_target_content_init(
            image_org, image_tgt, local_eps_map, topk_mask, blend=tci_blend)

        topk_mask_for_loss = topk_mask
        target_px_weight   = tci_px_w
        print(f"[Inject] Target-content warm-start: top {tci_ratio:.0%} UBMB boxes, "
              f"blend={tci_blend}, px_weight={tci_px_w}")
    else:
        topk_mask_for_loss = None
        target_px_weight   = 0.0
        delta_init         = None
        print("[Inject] Target-content injection disabled (target_inject_enable=false)")

    save_json({
        "estimated_patch": int(est_patch),
        "routing_patch": int(routing_patch),
        "routing_info": info,
        "foa_topk_patches": int(topk_k),
        "foa_topk_adaptive_ubmb": bool(use_adaptive_topk),
        "foa_topk_ubmb_aggregate": str(topk_agg),
        "ubmb_token_topk": bool(ubmb_tok_enable),
        "ubmb_topk_blend": float(blend_ubmb),
        "source_label": source_label,
        "target_label": target_label,
        "base_text": base_text,
        "mask_mode": mask_mode,
        "surrogate_weights": surrogate_info,
        "target_inject_enable": bool(tci_enable),
        "target_inject_ratio": float(tci_ratio),
        "target_inject_blend": float(tci_blend),
        "target_inject_px_weight": float(tci_px_w),
    }, os.path.join(out_dir, "routing_summary.json"))

    adv, delta = soft_local_eps_foa_attack(
        cfg, ensemble_extractor, ensemble_loss,
        source_crop, target_crop, image_org, image_tgt, gw, ef,
        model_weights=surrogate_weights,
        topk_spatial_mask=topk_mask_for_loss,
        target_pixel_weight=target_px_weight,
        delta_init=delta_init,
        surrogate_info=surrogate_info)

    if hasattr(ensemble_extractor, "set_ubmb_token_priority"):
        ensemble_extractor.set_ubmb_token_priority(None)
    if hasattr(ensemble_loss, "set_ubmb_token_priority"):
        ensemble_loss.set_ubmb_token_priority(None)

    noise_stats = save_noise_visualization(delta, os.path.join(out_dir, "adversarial_noise.png"))
    print(f"[Noise] L∞: {noise_stats['linf_norm']:.2f}, L2(norm): {noise_stats['l2_norm_normalized']:.4f}, PSNR: {noise_stats['psnr_db']:.1f}dB")
    
    if run_comparison:
        print("\n[Comparison] Creating comparison report...")
        adv_pil = tensor_to_pil((adv * 255).clamp(0, 255))
        report = create_comparison_report(src_pil, adv_pil, tgt_pil, delta,
                                         qwen, BASE_PROMPT, out_dir,
                                         source_label, target_label)
        print(f"[Report] Attack success: {report['attack_success']}")
    
    return adv, info

@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    seed_everything(2023)
    setup_wandb(cfg, tags=["adaptive_ubmb_sinkhorn", "qwen_blackbox"])
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    
    models = []
    for bb in cfg.model.backbone:
        key = str(bb).lower().strip()
        if key not in BACKBONE_MAP:
            raise ValueError(f"Unknown backbone '{bb}'. Available: {list(BACKBONE_MAP.keys())}")
        factory = BACKBONE_MAP[key]
        m = (factory() if callable(factory) else factory).eval().to(cfg.model.device).requires_grad_(False)
        models.append(m)
        ps = getattr(m, "native_patch_size", "?")
        print(f"[Model] {key} (native_patch={ps}px)")
    
    topk_k = int(getattr(getattr(cfg, "model", None), "topk_patches", 64))
    ensemble_extractor = EnsembleFeatureExtractor_ot_topk(models, cluster_number=topk_k)
    ensemble_loss      = EnsembleFeatureLoss_OT_topk_patches(models, cluster_number=topk_k)
    
    transform_fn = transforms.Compose([
        transforms.Resize(cfg.model.input_res, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.model.input_res),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(to_tensor_255),
    ])
    
    clean_data = FlatImageDataset(cfg.data.cle_data_path, transform=transform_fn)
    target_data = FlatImageDataset(cfg.data.tgt_data_path, transform=transform_fn)
    loader_clean = torch.utils.data.DataLoader(clean_data, batch_size=cfg.data.batch_size, shuffle=False)
    loader_target = torch.utils.data.DataLoader(target_data, batch_size=cfg.data.batch_size, shuffle=False)

    src_keywords_index = load_keywords_index(cfg.data.cle_data_path)
    tgt_keywords_index = load_keywords_index(cfg.data.tgt_data_path)
    src_auto_labels = load_auto_labels_cache(cfg.data.cle_data_path)
    tgt_auto_labels = load_auto_labels_cache(cfg.data.tgt_data_path)
    src_auto_dirty = False
    tgt_auto_dirty = False
    print(f"[Labels] source keywords: {len(src_keywords_index)} entries from {cfg.data.cle_data_path}")
    print(f"[Labels] target keywords: {len(tgt_keywords_index)} entries from {cfg.data.tgt_data_path}")
    
    _cs = (getattr(cfg.model, "crop_scale_1", None) or getattr(cfg.model, "crop_scale", None) or (0.5, 1.0))
    crop_scale = tuple(float(x) for x in _cs)
    use_source_crop = bool(getattr(cfg.model, "use_source_crop", False))
    use_target_crop = bool(getattr(cfg.model, "use_target_crop", False))
    source_crop = (transforms.RandomResizedCrop(cfg.model.input_res, scale=crop_scale)
                   if use_source_crop else torch.nn.Identity())
    target_crop = (transforms.RandomResizedCrop(cfg.model.input_res, scale=crop_scale)
                   if use_target_crop else torch.nn.Identity())
    
    config_hash = hash_training_config(cfg)
    out_root = os.path.join(cfg.data.output, "img", config_hash + "_target_noise_masking")
    ensure_dir(out_root)
    print(f"[Output] {out_root}")
    
    vlm = build_vlm_blackbox(cfg)
    print(f"[Blackbox] UBMB queries via: {getattr(getattr(cfg, 'blackbox', None), 'model_name', 'gpt5')}")
    _atk = getattr(cfg, "attack", None)
    config_source_label = getattr(_atk, "source_label", "")
    config_target_label = getattr(_atk, "target_label", "")
    mask_mode    = getattr(_atk, "mask_mode", MASK_MODE)
    if not config_source_label and not src_keywords_index and not src_auto_labels:
        print("[Labels] No source metadata found — will query VLM once per source image before UBMB.")
    if not config_target_label and not tgt_keywords_index and not tgt_auto_labels:
        print("[Labels] No target metadata found — will query VLM once per target image before UBMB.")
    
    print(f"\n[Config] Mask Mode: {mask_mode}")
    print(f"[Config] Source crop (attack forward): {use_source_crop}")
    print(f"[Config] Target crop (OT ground truth): {use_target_crop} "
          f"({'random sub-crop' if use_target_crop else 'full resized target image'})")
    if config_source_label:
        print(f"[Config] Source label fallback (config): {config_source_label}")
    if config_target_label:
        print(f"[Config] Target label fallback (config): {config_target_label}")
    
    for i, ((img_org, _, path_org), (img_tgt, _, path_tgt)) in enumerate(zip(loader_clean, loader_target)):
        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break
        print(f"\n{'='*50} Batch {i+1} {'='*50}")
        img_org = img_org.to(cfg.model.device)
        img_tgt = img_tgt.to(cfg.model.device)
        
        for b in range(img_org.shape[0]):
            pth = path_org[b]
            tgt_pth = path_tgt[b]
            src_folder = os.path.basename(os.path.dirname(pth)) or "images"
            src_name = os.path.splitext(os.path.basename(pth))[0]
            sample_dir = os.path.join(out_root, f"sample_{i:05d}_{b:02d}")
            ensure_dir(sample_dir)

            src_pil = tensor_to_pil(img_org[b:b+1])
            tgt_pil = tensor_to_pil(img_tgt[b:b+1])
            source_label, source_label_src = resolve_image_label(
                pth, src_keywords_index, src_auto_labels,
                config_fallback=config_source_label, qwen=vlm, pil_image=src_pil,
                allow_vlm=True)
            target_label, target_label_src = resolve_image_label(
                tgt_pth, tgt_keywords_index, tgt_auto_labels,
                config_fallback=config_target_label, qwen=vlm, pil_image=tgt_pil,
                allow_vlm=True)
            if source_label_src == "vlm":
                src_auto_dirty = True
            if target_label_src == "vlm":
                tgt_auto_dirty = True
            print(f"[Labels] {src_name}: source={source_label!r} ({source_label_src}) "
                  f"target={target_label!r} ({target_label_src})")
            if not source_label or not target_label:
                print("[WARN] Missing source/target label — UBMB semantic/CP/IOS signals may be weak.")
            save_json({
                "source_image": pth,
                "target_image": tgt_pth,
                "source_label": source_label,
                "target_label": target_label,
                "source_label_source": source_label_src,
                "target_label_source": target_label_src,
            }, os.path.join(sample_dir, "image_labels.json"))

            run_comparison = (i == 0 and b == 0)
            
            adv_b, info = dynamic_ubmb_sinkhorn_attack_one(
                cfg, ensemble_extractor, ensemble_loss, source_crop, target_crop,
                img_org[b:b+1], img_tgt[b:b+1], vlm, out_dir=sample_dir,
                source_label=source_label, target_label=target_label,
                mask_mode=mask_mode, run_comparison=run_comparison)
            
            save_dir = os.path.join(out_root, src_folder)
            ensure_dir(save_dir)
            save_path = os.path.join(save_dir, src_name + ".png")
            torchvision.utils.save_image(adv_b[0].detach().cpu(), save_path)
            print(f"[SAVE] {save_path}")

    if src_auto_dirty:
        save_auto_labels_cache(cfg.data.cle_data_path, src_auto_labels)
        print(f"[Labels] Saved source auto_labels cache → {_auto_labels_cache_path(cfg.data.cle_data_path)}")
    if tgt_auto_dirty:
        save_auto_labels_cache(cfg.data.tgt_data_path, tgt_auto_labels)
        print(f"[Labels] Saved target auto_labels cache → {_auto_labels_cache_path(cfg.data.tgt_data_path)}")
    
    wandb.finish()
    print("\n[DONE] All outputs saved!")
    print(f"[DONE] Check {out_root} for results")

    _cmp = getattr(cfg, "compare", None)
    foa_adv = getattr(_cmp, "foa_adv_dir", None) if _cmp else None
    m_adv   = getattr(_cmp, "m_adv_dir",   None) if _cmp else None

    your_adv = os.path.join(out_root, "nips17")
    if not os.path.isdir(your_adv):
        your_adv = out_root

    print("\n[Compare] Running perturbation comparison vs baselines...")
    run_baseline_comparison(
        your_adv_dir=your_adv,
        clean_dir=cfg.data.cle_data_path,
        foa_adv_dir=foa_adv,
        m_adv_dir=m_adv,
        out_dir=os.path.join(out_root, "baseline_comparison"),
    )

if __name__ == "__main__":
    main()