from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from hydra.core.config_store import ConfigStore

@dataclass
class AttackConfig:
    name: str = "attention_guided"
    source_label: str = ""
    target_label: str = ""
    mask_mode: str = "gray"

@dataclass
class WandbConfig:
    entity: str = "???"
    project: str = "local_adversarial_attack"

@dataclass
class BlackboxConfig:
    model_name: str = "qwen"
    batch_size: int = 1
    timeout: int = 30
    api_keys_file: str = "api_keys.yaml"
    openai_model_id: Optional[str] = None
    anthropic_model_id: Optional[str] = None
    thinking_budget_tokens: int = 1024
    max_completion_tokens: int = 80

@dataclass
class DataConfig:
    batch_size: int = 1
    num_samples: int = 1000
    cle_data_path: str = "resources/images/bigscale"
    tgt_data_path: str = "resources/images/target_images"
    output: str = "./Ours"

@dataclass
class OptimConfig:
    alpha: float = 1.0
    epsilon: int = 8
    steps: int = 300
    target_refresh_every: int = 10
    surrogate_sample_ratio: float = 1.0
    use_input_diversity: bool = True
    dim_prob: float = 0.5
    dim_resize_range: Tuple[float, float] = (0.8, 1.0)
    use_ti_grad: bool = True
    ti_kernel_size: int = 5
    ti_sigma: float = 1.0
    use_scale_invariant: bool = True
    si_scales: List[float] = (1.0, 0.5, 0.25)
    warmup_steps: int = 50
    attack_method: str = "mifgsm"
    global_loss_weight: float = 0.0
    use_mca: bool = True
    mca_num_crops: int = 4
    use_momentum: bool = True
    momentum_decay: float = 0.9
    use_adaptive_gw: bool = True
    adaptive_gw_ema: float = 0.9
    adaptive_gw_blend: float = 0.5
    use_paradigm_curriculum: bool = True
    paradigm_curriculum_strength: float = 1.0
    patch_ot_weight: float = 0.1

@dataclass
class ModelConfig:
    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = False
    crop_scale: Tuple[float, float] = (0.5, 0.9)
    ensemble: bool = True
    device: str = "cuda:0"
    backbone: List[str] = ("L336", "B16", "B32", "Laion")
    patch_size: int = 32
    top_k_ratio: float = 0.2
    attention_labels: List[str] = ("object", "main subject", "foreground")
    drift_weight: float = 0.5
    use_response_drift: bool = True
    patch_surrogate_tolerance: Optional[int] = None
    topk_patches: int = 64
    adaptive_topk_ubmb: bool = True
    topk_patches_min: int = 10
    topk_ubmb_aggregate: str = "mean"
    ubmb_token_topk: bool = True
    ubmb_topk_blend: float = 0.75
    ubmb_topk_box_score: str = "product"
    patch_weight_temperature: float = 0.4
    qwen_local_path: Optional[str] = None
    use_fast_drift: bool = False
    fast_drift_masks: int = 15

    attention_similarity_blend: float = 1.0

    target_ubmb_topk: bool = True
    num_target_ubmb_masks: int = 15
    target_ubmb_top_k_ratio: float = 0.25

    target_copy_enable: bool = True
    target_copy_min_weight: float = 0.08
    target_copy_weight: float = 2.0
    target_copy_project_blend: float = 0.55

    target_inject_enable: bool = False
    target_inject_ratio: float = 0.35
    target_inject_blend: float = 0.95
    target_inject_px_weight: float = 0.25

@dataclass
class MainConfig:
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandbConfig = WandbConfig()
    blackbox: BlackboxConfig = BlackboxConfig()
    attack: AttackConfig = field(default_factory=AttackConfig)

@dataclass
class Ensemble3ModelsConfig(MainConfig):
    data: DataConfig = DataConfig(batch_size=1)
    model: ModelConfig = ModelConfig(
        use_source_crop=True,
        use_target_crop=False,
        backbone=["B16", "B32", "Laion"],
    )

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
cs.store(name="ensemble_3models", node=Ensemble3ModelsConfig)