
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from multi_vlm_eval import (
    DEFAULT_API_KEYS_FILE,
    CAPTION_PROMPT_SHORT,
    CAPTION_PROMPT_GEMINI,
    encode_image_b64,
    load_api_keys_file,
    resolve_keys,
    sorted_images,
    load_keywords_json,
    DEFAULT_KEYWORDS_JSON,
    GPTScorer,
    local_similarity,
)

from PIL import Image

DEFAULT_TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]
DEFAULT_THRESHOLDS   = [0.1, 0.2, 0.3, 0.4, 0.5]

DEFAULT_MODELS = [
    "claude3.5",
    "claude3.7",
    "gpt4o",
    "gpt4.1",
    "gemini2.5",
    "o3",
    "claude3.7-think",
    "gemini2.5flash",
]

def _caption_claude(image_path: str, model_id: str, api_key: str,
                    temperature: float, thinking: bool = False) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    b64    = encode_image_b64(image_path)
    media  = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    kw: dict = dict(
        model=model_id, max_tokens=400,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": CAPTION_PROMPT_SHORT},
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
        ]}],
    )
    if thinking:
        kw["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        kw["max_tokens"] = 1600
        kw["temperature"] = 1.0
    else:
        kw["temperature"] = min(1.0, max(0.0, temperature))
    resp = client.messages.create(**kw)
    content = resp.content
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text
    return content[0].text if content else ""

def _caption_openai(image_path: str, model_id: str, api_key: str,
                    temperature: float, reasoning_effort: Optional[str] = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    b64    = encode_image_b64(image_path)
    kw: dict = dict(
        model=model_id,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": CAPTION_PROMPT_SHORT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    )
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
        kw["max_completion_tokens"] = 5000
    else:
        kw["temperature"] = temperature
        kw["max_tokens"]  = 150
    resp = client.chat.completions.create(**kw)
    return resp.choices[0].message.content.strip()

def _caption_gemini(image_path: str, model_id: str, api_key: str,
                    temperature: float) -> str:
    from google import genai
    from google.genai import types as gtypes
    client = genai.Client(api_key=api_key)
    image  = Image.open(image_path)
    config = gtypes.GenerateContentConfig(temperature=temperature, max_output_tokens=150)
    resp   = client.models.generate_content(
        model=model_id,
        contents=[CAPTION_PROMPT_GEMINI, image],
        config=config,
    )
    return resp.text.strip()

def caption_with_temp(model_name: str, image_path: str, temperature: float,
                      keys: Dict[str, str]) -> str:
    n = model_name.lower()
    for attempt in range(4):
        try:
            if n == "claude3.5":
                return _caption_claude(image_path, "claude-3-5-sonnet-20241022",
                                       keys["anthropic"], temperature)
            if n == "claude3.7":
                return _caption_claude(image_path, "claude-3-7-sonnet-20250219",
                                       keys["anthropic"], temperature)
            if n in ("claude3.7-think", "claude3.7thinking"):
                return _caption_claude(image_path, "claude-3-7-sonnet-20250219",
                                       keys["anthropic"], temperature, thinking=True)
            if n == "gpt4o":
                return _caption_openai(image_path, "gpt-4o", keys["openai"], temperature)
            if n in ("gpt4.1", "gpt-4.1"):
                return _caption_openai(image_path, "gpt-4.1", keys["openai"], temperature)
            if n == "o3":
                return _caption_openai(image_path, "o3", keys["openai"],
                                       temperature, reasoning_effort="medium")
            if n in ("gemini2.5", "gemini2.5pro"):
                return _caption_gemini(image_path, "gemini-2.5-pro-preview-05-06",
                                       keys["google"], temperature)
            if n == "gemini2.5flash":
                return _caption_gemini(image_path, "gemini-2.5-flash-preview-04-17",
                                       keys["google"], temperature)
            raise ValueError(f"Unknown model for temp sweep: {model_name!r}")
        except Exception as e:
            err = str(e)
            if any(kw in err.lower() for kw in ["billing", "credit", "quota", "insufficient"]):
                return "[BILLING_ERROR]"
            if attempt == 3:
                return f"[ERROR: {e}]"
            time.sleep(min(2 ** attempt, 20))
    return "[ERROR: max retries]"

def run_sweep(
    adv_dir: str,
    target_dir: Optional[str],
    model_names: List[str],
    temperatures: List[float],
    thresholds: List[float],
    keys: Dict[str, str],
    out_dir: str,
    max_images: int = 50,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    adv_files = sorted_images(adv_dir)[:max_images]
    tgt_files = sorted_images(target_dir)[:max_images] if target_dir else []
    n_imgs    = len(adv_files)
    if n_imgs == 0:
        print(f"[Sweep] No images found in {adv_dir}")
        return
    print(f"[Sweep] {n_imgs} adversarial images  |  models: {model_names}")
    print(f"[Sweep] Temperatures: {temperatures}")
    print(f"[Sweep] Thresholds:   {thresholds}")

    scorer = GPTScorer(keys.get("openai", ""))

    results: Dict[str, Dict[float, List[float]]] = {
        m: {t: [] for t in temperatures} for m in model_names
    }
    tgt_cache: Dict[Tuple[str, int], str] = {}
    raw_rows: List[dict] = []

    for img_idx, adv_path in enumerate(adv_files):
        tgt_path = tgt_files[img_idx] if img_idx < len(tgt_files) else None
        print(f"\n  [img {img_idx+1}/{n_imgs}] {os.path.basename(adv_path)}")

        for model_name in model_names:
            for temp in temperatures:
                print(f"    {model_name:20s}  T={temp:.1f} ...", end=" ", flush=True)
                adv_cap = caption_with_temp(model_name, adv_path, temp, keys)
                if "[BILLING_ERROR]" in adv_cap:
                    print("BILLING — skipping model")
                    break

                tgt_cap = ""
                if tgt_path:
                    ck = (model_name, img_idx, temp)
                    if ck not in tgt_cache:
                        tgt_cache[ck] = caption_with_temp(model_name, tgt_path, temp, keys)
                    tgt_cap = tgt_cache[ck]

                sim = scorer.compute_similarity(tgt_cap, adv_cap) if tgt_cap else 0.0
                results[model_name][temp].append(sim)
                print(f"sim={sim:.3f}")

                raw_rows.append({
                    "model": model_name, "img_idx": img_idx,
                    "temperature": temp,
                    "adv_caption": adv_cap, "target_caption": tgt_cap,
                    "sim_score": round(sim, 4),
                })

    raw_csv = os.path.join(out_dir, "temp_threshold_results.csv")
    if raw_rows:
        with open(raw_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader(); w.writerows(raw_rows)
        print(f"\n[Saved] {raw_csv}")

    asr: Dict[str, Dict[float, Dict[float, float]]] = {}
    for model_name in model_names:
        asr[model_name] = {}
        for temp in temperatures:
            sims = results[model_name][temp]
            asr[model_name][temp] = {}
            for thresh in thresholds:
                n = len(sims)
                asr[model_name][temp][thresh] = (
                    sum(1 for s in sims if s >= thresh) / n if n else 0.0
                )

    _plot_heatmaps(asr, model_names, temperatures, thresholds, out_dir)

def _plot_heatmaps(asr, model_names, temperatures, thresholds, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n_models = len(model_names)
        ncols    = min(4, n_models)
        nrows    = (n_models + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.2, nrows * 3.5),
                                 squeeze=False)
        fig.suptitle("ASR-sim: Temperature × Similarity Threshold",
                     fontsize=13, fontweight="bold", y=1.01)

        for mi, model_name in enumerate(model_names):
            ax  = axes[mi // ncols][mi % ncols]
            mat = np.array([[asr[model_name][t][th]
                             for th in thresholds]
                            for t  in temperatures])

            im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(thresholds)))
            ax.set_yticks(range(len(temperatures)))
            ax.set_xticklabels([f"{th:.2f}" for th in thresholds], fontsize=8)
            ax.set_yticklabels([f"{t:.1f}"  for t  in temperatures], fontsize=8)
            ax.set_xlabel("Threshold", fontsize=8)
            ax.set_ylabel("Temperature", fontsize=8)
            ax.set_title(model_name, fontsize=9, fontweight="bold")

            for ri in range(len(temperatures)):
                for ci in range(len(thresholds)):
                    v = mat[ri, ci]
                    ax.text(ci, ri, f"{v:.0%}", ha="center", va="center",
                            fontsize=8, color="black" if 0.3 < v < 0.8 else "white")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig_i, ax_i = plt.subplots(figsize=(5, 4))
            im_i = ax_i.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
            ax_i.set_xticks(range(len(thresholds)))
            ax_i.set_yticks(range(len(temperatures)))
            ax_i.set_xticklabels([f"{th:.2f}" for th in thresholds])
            ax_i.set_yticklabels([f"{t:.1f}" for t in temperatures])
            ax_i.set_xlabel("Similarity Threshold")
            ax_i.set_ylabel("Temperature")
            ax_i.set_title(f"ASR — {model_name}", fontweight="bold")
            for ri in range(len(temperatures)):
                for ci in range(len(thresholds)):
                    v = mat[ri, ci]
                    ax_i.text(ci, ri, f"{v:.0%}", ha="center", va="center",
                              fontsize=9, color="black" if 0.3 < v < 0.8 else "white")
            plt.colorbar(im_i, ax=ax_i)
            fig_i.tight_layout()
            safe = model_name.replace("/", "-").replace(" ", "_")
            fig_i.savefig(os.path.join(out_dir, f"temp_threshold_heatmap_{safe}.png"),
                          dpi=150, bbox_inches="tight")
            plt.close(fig_i)

        for mi in range(n_models, nrows * ncols):
            axes[mi // ncols][mi % ncols].set_visible(False)

        fig.tight_layout()
        grid_path = os.path.join(out_dir, "temp_threshold_asr_grid.png")
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {grid_path}")

        default_t = 1.0 if 1.0 in temperatures else temperatures[-1]
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        cmap_l = plt.get_cmap("tab10")
        for mi, model_name in enumerate(model_names):
            vals = [asr[model_name][default_t][th] for th in thresholds]
            ax2.plot(thresholds, vals, marker="o", label=model_name,
                     color=cmap_l(mi / max(len(model_names), 1)))
        ax2.set_xlabel("Similarity Threshold")
        ax2.set_ylabel("ASR-sim")
        ax2.set_title(f"ASR vs Threshold (T={default_t:.1f})", fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        line_path = os.path.join(out_dir, "asr_vs_threshold_line.png")
        fig2.savefig(line_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"[Saved] {line_path}")

        default_thresh = 0.3 if 0.3 in thresholds else thresholds[0]
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        for mi, model_name in enumerate(model_names):
            vals = [asr[model_name][t][default_thresh] for t in temperatures]
            ax3.plot(temperatures, vals, marker="s", label=model_name,
                     color=cmap_l(mi / max(len(model_names), 1)))
        ax3.set_xlabel("Temperature")
        ax3.set_ylabel("ASR-sim")
        ax3.set_title(f"ASR vs Temperature (threshold={default_thresh:.2f})",
                      fontweight="bold")
        ax3.set_ylim(0, 1.05)
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(alpha=0.3)
        fig3.tight_layout()
        temp_path = os.path.join(out_dir, "asr_vs_temperature_line.png")
        fig3.savefig(temp_path, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"[Saved] {temp_path}")

    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")
        import traceback; traceback.print_exc()

def parse_args():
    p = argparse.ArgumentParser(
        description="Temperature × threshold ASR sweep for frontier VLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adv_dir",     required=True,
                   help="Directory of adversarial images (PNG/JPG).")
    p.add_argument("--target_dir",  default=None,
                   help="Directory of target images (for similarity). "
                        "If omitted similarity is 0 for all images.")
    p.add_argument("--models",      nargs="+", default=DEFAULT_MODELS,
                   help=f"Models to sweep. Default: {DEFAULT_MODELS}")
    p.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    p.add_argument("--thresholds",   nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--max_images",   type=int,  default=50,
                   help="Cap number of adversarial images evaluated (keeps API cost low).")
    p.add_argument("--out_dir",      default="eval/temp_threshold_results")
    p.add_argument("--api_keys_file", default=DEFAULT_API_KEYS_FILE)
    return p.parse_args()

def main():
    args = parse_args()
    keys = resolve_keys(args.api_keys_file)
    print(f"[Keys] openai={'yes' if keys['openai'] else 'NO'}  "
          f"anthropic={'yes' if keys['anthropic'] else 'NO'}  "
          f"google={'yes' if keys['google'] else 'NO'}")
    run_sweep(
        adv_dir     = args.adv_dir,
        target_dir  = args.target_dir,
        model_names = args.models,
        temperatures = args.temperatures,
        thresholds  = args.thresholds,
        keys        = keys,
        out_dir     = args.out_dir,
        max_images  = args.max_images,
    )

if __name__ == "__main__":
    main()
