from .Base import BaseFeatureExtractor, find_hf_cache_for_model
from torchvision import transforms
from transformers import Dinov2Model
from typing import Optional, List

import torch

MODEL_DICT = {
    "dino_small": "facebook/dinov2-small",
    "dino_base": "facebook/dinov2-base",
    "dino_large": "facebook/dinov2-large",
}

class Dinov2FeatureExtractor(BaseFeatureExtractor):

    native_patch_size: int = 14

    def __init__(
        self,
        model_name: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super(Dinov2FeatureExtractor, self).__init__()
        hf_id = MODEL_DICT[model_name]
        self.model = Dinov2Model.from_pretrained(
            hf_id, cache_dir=find_hf_cache_for_model(hf_id))
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model(**inputs)
        pooler = outputs.pooler_output
        return pooler / (pooler.norm(dim=1, keepdim=True) + 1e-8)

    def global_local_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model(**inputs, output_hidden_states=False)
        hidden = outputs.last_hidden_state
        global_feature = hidden[:, 0, :]
        global_feature = global_feature / (global_feature.norm(dim=1, keepdim=True) + 1e-8)
        local_feature = hidden[:, 1:, :]
        local_feature = local_feature / (local_feature.norm(dim=-1, keepdim=True) + 1e-8)
        return global_feature, local_feature
