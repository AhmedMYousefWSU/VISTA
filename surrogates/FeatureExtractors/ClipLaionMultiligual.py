import os
import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from .Base import BaseFeatureExtractor
from torchvision import transforms

_DEFAULT_MULTILINGUAL_CLIP = os.environ.get(
    "CLIP_MULTILINGUAL_HF_ID", "openai/clip-vit-base-patch32"
)

class ClipLaionMultiligualFeatureExtractor(BaseFeatureExtractor):
    native_patch_size: int = 32
    def __init__(self):
        super(ClipLaionMultiligualFeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained(_DEFAULT_MULTILINGUAL_CLIP)
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def global_local_features(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        hidden = outputs.last_hidden_state
        global_feature = hidden[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = hidden[:, 1:, :]
        local_feature = local_feature / (local_feature.norm(dim=-1, keepdim=True) + 1e-8)
        return global_feature, local_feature
