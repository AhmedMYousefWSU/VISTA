import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel
from .Base import BaseFeatureExtractor, find_hf_cache_for_model

def _build_normalizer(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

class _Siglip2Base(BaseFeatureExtractor):

    _ckpt: str = ""
    _size: int = 224
    native_patch_size: int = 16

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(self._ckpt, cache_dir=find_hf_cache_for_model(self._ckpt))
        self.normalizer = _build_normalizer(self._size)

    def _vision_hidden(self, x: torch.Tensor):
        pv = self.normalizer(x).to(next(self.model.parameters()).device)
        out = self.model.vision_model(pixel_values=pv)
        return out.last_hidden_state.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pv = self.normalizer(x).to(next(self.model.parameters()).device)
        feat = self.model.get_image_features(pixel_values=pv).float()
        return feat / (feat.norm(dim=1, keepdim=True) + 1e-8)

    def global_local_features(self, x: torch.Tensor):
        hidden = self._vision_hidden(x)
        g = hidden.mean(dim=1)
        g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
        l = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-8)
        return g, l

class Siglip2B16FeatureExtractor(_Siglip2Base):
    _ckpt = "google/siglip2-base-patch16-224"
    _size = 224
    native_patch_size = 16

class Siglip2B32FeatureExtractor(_Siglip2Base):
    _ckpt = "google/siglip2-base-patch32-256"
    _size = 256
    native_patch_size = 32

class Siglip2G16FeatureExtractor(_Siglip2Base):
    _ckpt = "google/siglip2-giant-opt-patch16-256"
    _size = 256
    native_patch_size = 16
