import torch
from transformers import CLIPModel
from .Base import BaseFeatureExtractor, find_hf_cache_for_model
from torchvision import transforms

class ClipLaionB32FeatureExtractor(BaseFeatureExtractor):

    native_patch_size: int = 32

    def __init__(self):
        super(ClipLaionB32FeatureExtractor, self).__init__()
        _hf_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        self.model = CLIPModel.from_pretrained(
            _hf_id, cache_dir=find_hf_cache_for_model(_hf_id))
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        return image_features / image_features.norm(dim=1, keepdim=True)

    def global_local_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        features = outputs.last_hidden_state
        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / (local_feature.norm(dim=-1, keepdim=True) + 1e-8)
        return global_feature, local_feature
