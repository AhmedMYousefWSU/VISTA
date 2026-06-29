import torch
from transformers import ViTModel
from .Base import BaseFeatureExtractor, find_hf_cache_for_model
from torchvision import transforms

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class VisionTransformerFeatureExtractor(BaseFeatureExtractor):
    native_patch_size: int = 16
    def __init__(self):
        super(VisionTransformerFeatureExtractor, self).__init__()
        _hf_id = "google/vit-base-patch16-224-in21k"
        self.model = ViTModel.from_pretrained(
            _hf_id, cache_dir=find_hf_cache_for_model(_hf_id))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD), 
        ]
    )

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        outputs_features = self.model(**inputs)
        image_features = outputs_features.pooler_output
        image_features = image_features / image_features.norm(dim=1, keepdim=True) 
        return image_features

    def global_local_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        global_feature = hidden[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = hidden[:, 1:, :]
        local_feature = local_feature / (local_feature.norm(dim=-1, keepdim=True) + 1e-8)
        return global_feature, local_feature
