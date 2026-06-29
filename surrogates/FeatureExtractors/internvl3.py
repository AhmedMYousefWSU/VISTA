import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageTextToText
from .Base import BaseFeatureExtractor, find_hf_cache_for_model

_DEFAULT_INTERNVL_PATH = os.environ.get(
    "INTERNVL3_MODEL_PATH",
    "OpenGVLab/InternVL3-1B-hf",
)

_INTERNVL_INPUT_SIZE = 448

class InternVL3FeatureExtractor(BaseFeatureExtractor):

    native_patch_size: int = 14

    def __init__(
        self,
        model_path: str = _DEFAULT_INTERNVL_PATH,
    ):
        super().__init__()
        cache = None if os.path.isdir(model_path) else find_hf_cache_for_model(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=cache,
        )
        self.normalizer = transforms.Compose([
            transforms.Resize(
                (_INTERNVL_INPUT_SIZE, _INTERNVL_INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def _pixel_values(self, x: torch.Tensor) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        return self.normalizer(x).to(device=dev, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pv = self._pixel_values(x)
        feat = self.model.get_image_features(pixel_values=pv).float()
        return feat / (feat.norm(dim=1, keepdim=True) + 1e-8)

    def global_local_features(self, x: torch.Tensor):
        pv = self._pixel_values(x)
        hidden = self.model.vision_tower(pixel_values=pv).last_hidden_state.float()
        g = hidden[:, 0, :]
        g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
        l = hidden[:, 1:, :]
        l = l / (l.norm(dim=-1, keepdim=True) + 1e-8)
        return g, l
