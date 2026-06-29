# MIRROR Attack

**MIRROR** is a black-box adversarial attack against Vision-Language Models (VLMs). It uses a UBMB (Unified Black-Box Masking-Based) attribution method to identify semantically important image regions, then routes gradient updates through an ensemble of surrogate vision encoders (CLIP, BLIP, DINOv2, InternVL3, SigLIP2) to generate imperceptible perturbations that shift VLM descriptions toward a target concept.

## Requirements

- NVIDIA GPU (CUDA required)
- Python 3.9+
- A local Qwen3-VL-8B-Instruct model **or** an OpenAI API key (GPT-4o / GPT-5)

## Installation

```bash
git clone https://github.com/AhmedMYousefWSU/MIRROR.git
cd MIRROR

conda create -n mirror python=3.10 -y
conda activate mirror

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> The CLIP package installs from GitHub. Make sure `git` is available.

## Data Layout

Place images in the following structure before running:

```
resources/images/
├── clean/          # source images to perturb
└── target/         # target images (one per source, paired by sort order)
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`

Optionally add a `keywords.json` inside either folder to provide semantic labels:

```json
[
  {"image": "cat.jpg", "keywords": ["cat", "tabby"]},
  {"image": "dog.jpg", "keywords": ["dog", "golden retriever"]}
]
```

## Black-Box VLM Setup

MIRROR queries a VLM for each image during the UBMB attribution phase. Two backends are supported:

### Option A — Local Qwen3-VL (default)

Download the model and point to it via an environment variable:

```bash
export QWEN3_VL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

If the variable is not set, the code looks for `Qwen3-VL-8B-Instruct/` inside the repo root.

### Option B — OpenAI GPT-4o / GPT-5

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

Or create `api_keys.yaml` in the repo root:

```yaml
gpt5: sk-...
# gpt4o: sk-...   # uncomment to use GPT-4o instead
```

Then update the config (see below) to use the API model:

```yaml
blackbox:
  model_name: "gpt5"   # or "gpt4o"
```

## Running the Attack

```bash
python mirror_attack.py
```

This uses the default config at [config/ensemble_3models.yaml](config/ensemble_3models.yaml). Outputs are written to `./mirror_attack_output/`.

### Override config values on the command line (Hydra syntax)

```bash
# Change source/target image directories
python mirror_attack.py data.cle_data_path=my_images/clean data.tgt_data_path=my_images/target

# Limit to 10 images
python mirror_attack.py data.num_samples=10

# Use a tighter perturbation budget (epsilon in pixel units, max 255)
python mirror_attack.py optim.epsilon=8

# Run more optimization steps
python mirror_attack.py optim.steps=500

# Change the black-box model to GPT-4o
python mirror_attack.py blackbox.model_name=gpt4o

# Provide semantic labels for source and target
python mirror_attack.py attack.source_label=cat attack.target_label=dog
```

### Key config options

| Parameter | Default | Description |
|---|---|---|
| `data.cle_data_path` | `resources/images/clean` | Source (clean) image folder |
| `data.tgt_data_path` | `resources/images/target` | Target image folder |
| `data.output` | `./mirror_attack_output` | Output root directory |
| `data.num_samples` | `100` | Number of images to process |
| `optim.epsilon` | `16` | L∞ perturbation budget (pixel units) |
| `optim.steps` | `300` | Optimization iterations per image |
| `optim.alpha` | `1.0` | Step size |
| `model.device` | `cuda:0` | GPU device |
| `model.backbone` | 9 models | Surrogate ensemble members |
| `blackbox.model_name` | `qwen` | VLM backend: `qwen`, `gpt4o`, `gpt5` |
| `attack.mask_mode` | `gray` | Mask style for UBMB: `gray`, `blur`, `target_noise`, `target_patch`, `target_blend` |

## Output Structure

```
mirror_attack_output/
└── img/
    └── <config_hash>_target_noise_masking/
        ├── <class_folder>/
        │   └── image_name.png          # adversarial image
        └── sample_00000_00/
            ├── image_labels.json       # source/target labels used
            ├── ubmb_patch_size_estimate.json
            ├── ubmb_scores_patch_64_gray.json
            ├── perturbation_noise_signed.png
            ├── perturbation_noise_magnitude.png
            ├── perturbation_delta.npy
            ├── perturbation_noise_stats.json
            ├── comparison_report.png
            └── detailed_report.json
```

## Evaluation

After generating adversarial examples, evaluate transfer across multiple VLMs:

```bash
# Edit paths inside the script or override via env variables
bash eval/run_multi_vlm_eval.sh

# Override the adversarial directory and output location
ADV_DIR_OVERRIDE=mirror_attack_output/img/<hash>/nips17 \
OUT_DIR=eval/results \
bash eval/run_multi_vlm_eval.sh
```

Supported evaluation models: `qwen`, `claude4.0`, `gpt4o`, `gpt5`.

API keys for evaluation models should be in `api_keys.yaml`:

```yaml
gpt5: sk-...
gpt4o: sk-...
```

## WandB Logging

Training metrics are logged to [Weights & Biases](https://wandb.ai) automatically. To disable:

```bash
wandb disabled
```

Or log in first:

```bash
wandb login
```

## Troubleshooting

**`No CUDA GPU detected`** — check `nvidia-smi` and ensure your PyTorch install matches your CUDA version.

**`Qwen model not found`** — set `QWEN3_VL_PATH` to the directory containing `config.json` for the Qwen model.

**`OpenAI API key not found`** — export `OPENAI_API_KEY` or add the key to `api_keys.yaml`.

**`salesforce-lavis` install fails** — install build tools first: `conda install -c conda-forge gcc gxx`.

## Acknowledgements

This project builds on the following prior works. We sincerely thank the authors for their outstanding contributions:

- **[M-Attack](https://github.com/VILA-Lab/M-Attack)** — the foundational surrogate-ensemble adversarial attack framework that MIRROR extends.
- **[FOA-Attack](https://github.com/jiaxiaojunQAQ/FOA-Attack)** — *Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment* (NeurIPS 2025), whose optimal-transport feature alignment loss is integrated into MIRROR's surrogate ensemble.
