
import argparse
import csv
import json
import math
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)

def stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def find_images_shallow(directory: str) -> Dict[str, str]:
    out = {}
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        ext = os.path.splitext(fn)[1].lower()
        if ext in VALID_EXT:
            out[os.path.splitext(fn)[0]] = os.path.join(directory, fn)
    return out

def perturbation_metrics(clean: np.ndarray, adv: np.ndarray) -> Dict[str, float]:
    if clean.shape != adv.shape:
        h, w = clean.shape[:2]
        adv = np.array(Image.fromarray(adv.astype(np.uint8)).resize(
            (w, h), Image.BICUBIC), dtype=np.float32)
    d = adv - clean
    n = float(d.size)
    l1   = float(np.abs(d).sum() / n / 255.0)
    l2   = float(math.sqrt((d ** 2).sum() / n) / 255.0)
    linf = float(np.abs(d).max())
    mse  = float((d ** 2).mean())
    psnr = float(20 * math.log10(255.0 / (math.sqrt(mse) + 1e-8)))
    return {"l1": l1, "l2": l2, "linf_255": linf, "psnr_db": psnr}

def save_noise_pngs(clean: np.ndarray, adv: np.ndarray,
                    out_dir: str, prefix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if clean.shape != adv.shape:
        h, w = clean.shape[:2]
        adv = np.array(Image.fromarray(adv.astype(np.uint8)).resize(
            (w, h), Image.BICUBIC), dtype=np.float32)
    delta = adv - clean

    signed = np.clip(delta * 10.0 + 128.0, 0, 255).astype(np.uint8)
    Image.fromarray(signed, "RGB").save(
        os.path.join(out_dir, f"{prefix}_noise_signed.png"))

    mag = np.sqrt((delta ** 2).sum(axis=2))
    mag_norm = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
    hot = cm.get_cmap("hot")
    mag_rgb = (hot(mag_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(mag_rgb, "RGB").save(
        os.path.join(out_dir, f"{prefix}_noise_magnitude.png"))

def save_noise_panel(samples: List[Tuple[str, np.ndarray, np.ndarray]],
                     out_path: str) -> None:
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, (method, clean, adv) in enumerate(samples):
        d = adv - clean
        signed = np.clip(d * 10.0 + 128.0, 0, 255).astype(np.uint8)
        axes[0][col].imshow(signed)
        axes[0][col].set_title(f"{method}\nSigned noise (10× amp)", fontsize=9)
        axes[0][col].axis("off")

        mag = np.sqrt((d ** 2).sum(axis=2))
        axes[1][col].imshow(mag, cmap="hot")
        axes[1][col].set_title(f"{method}\nMagnitude", fontsize=9)
        axes[1][col].axis("off")

    plt.suptitle("Per-method perturbation noise", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def normalise(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip())

def keyword_hit(text: str, keyword: str) -> bool:
    t = normalise(text)
    k = normalise(keyword)
    if k in t:
        return True
    for word in k.split():
        if re.search(r'\b' + re.escape(word) + r'\b', t):
            return True
    return False

def keyword_hits(text: str, keywords: List[str]) -> Dict[str, bool]:
    return {kw: keyword_hit(text, kw) for kw in keywords}

def evaluate_method(
    name: str,
    clean_images: Dict[str, str],
    adv_images: Dict[str, str],
    stems: List[str],
    keywords: List[str],
    noise_out_dir: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    rows = []
    for stem in stems:
        if stem not in clean_images or stem not in adv_images:
            continue
        c_arr = load_rgb(clean_images[stem])
        a_arr = load_rgb(adv_images[stem])

        metrics = perturbation_metrics(c_arr, a_arr)
        row = {"method": name, "stem": stem, **metrics}

        for kw in keywords:
            row[f"kw_{kw}_clean"] = None
            row[f"kw_{kw}_adv"]   = None

        rows.append(row)

        save_noise_pngs(c_arr, a_arr, noise_out_dir, f"{name.replace(' ','_')}_{stem}")

    if not rows:
        return rows, {}

    agg = {
        "method": name,
        "n": len(rows),
        "l1_mean":   float(np.mean([r["l1"]   for r in rows])),
        "l2_mean":   float(np.mean([r["l2"]   for r in rows])),
        "linf_mean": float(np.mean([r["linf_255"] for r in rows])),
        "psnr_mean": float(np.mean([r["psnr_db"]  for r in rows])),
    }
    return rows, agg

def bar_chart(agg_rows: List[Dict], metrics: List[Tuple[str, str]],
              title: str, out_path: str) -> None:
    names  = [r["method"] for r in agg_rows]
    n_met  = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(5 * n_met, 5))
    if n_met == 1:
        axes = [axes]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (label, key) in zip(axes, metrics):
        vals = [r.get(key, 0) for r in agg_rows]
        bars = ax.bar(names, vals, color=palette[:len(vals)], alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.3f}" if v < 10 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def asr_chart(agg_rows: List[Dict], keywords: List[str], out_path: str) -> None:
    names = [r["method"] for r in agg_rows]
    palette = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    w = 0.25
    for ki, kw in enumerate(keywords[:4]):
        vals = [r.get(f"asr_kw_{kw}", 0.0) for r in agg_rows]
        ax.bar(x + ki * w, vals, w, label=kw, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(names)
    ax.set_ylabel("ASR (fraction)")
    ax.set_title("Attack Success Rate per keyword", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def noise_grid(method_stem_pairs: List[Tuple[str, str, str]],
               clean_images: Dict[str, str], adv_dict_map: Dict[str, Dict],
               out_path: str, max_stems: int = 5) -> None:
    methods = list(dict.fromkeys(m for m, _, _ in method_stem_pairs))
    stems_used = list(dict.fromkeys(s for _, s, _ in method_stem_pairs))[:max_stems]
    nrows, ncols = len(methods), len(stems_used)
    if nrows == 0 or ncols == 0:
        return
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    for ri, method in enumerate(methods):
        for ci, stem in enumerate(stems_used):
            ax = axes[ri][ci]
            adv_images = adv_dict_map.get(method, {})
            if stem in clean_images and stem in adv_images:
                c = load_rgb(clean_images[stem])
                a = load_rgb(adv_images[stem])
                if c.shape != a.shape:
                    h, w = c.shape[:2]
                    a = np.array(Image.fromarray(a.astype(np.uint8)).resize(
                        (w, h), Image.BICUBIC), dtype=np.float32)
                mag = np.sqrt(((a - c) ** 2).sum(axis=2))
                ax.imshow(mag, cmap="hot")
            else:
                ax.set_facecolor("#333333")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="white", transform=ax.transAxes)
            ax.set_title(f"{method}\nimg {stem}", fontsize=7)
            ax.axis("off")

    plt.suptitle("Perturbation magnitude per method × image", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir",  required=True,
                    help="Directory of clean images (flat, named 0.png … 10.png).")
    ap.add_argument("--our_dir",    required=True,
                    help="Our method adv images (flat nips17/ subfolder).")
    ap.add_argument("--foa_dir",    required=True,
                    help="FOA-Attack adv images dir (flat).")
    ap.add_argument("--m_dir",      required=True,
                    help="M-Attack adv images dir (flat).")
    ap.add_argument("--our_sample_root", default=None,
                    help="Optional: our sample_XXXXX dirs root, for loading "
                         "perturbation_delta.npy + noise PNGs already saved.")
    ap.add_argument("--keywords",   default="",
                    help="Comma-separated keywords for ASR/keyword matching.")
    ap.add_argument("--stems",      default=None,
                    help="Comma-separated stems to compare (default: intersection).")
    ap.add_argument("--out_dir",    default="eval/three_way_results",
                    help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    noise_dir = os.path.join(args.out_dir, "noise_images")
    os.makedirs(noise_dir, exist_ok=True)

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    clean_map = find_images_shallow(args.clean_dir)
    our_map   = find_images_shallow(args.our_dir)
    foa_map   = find_images_shallow(args.foa_dir)
    m_map     = find_images_shallow(args.m_dir)

    print(f"[Load] clean={len(clean_map)}, ours={len(our_map)}, "
          f"foa={len(foa_map)}, m={len(m_map)}")

    if args.stems:
        forced = [s.strip() for s in args.stems.split(",") if s.strip()]
        all_stems = forced
    else:
        all_stems = sorted(
            set(clean_map) & set(our_map) & set(foa_map) & set(m_map),
            key=lambda x: int(x) if x.isdigit() else x)

    print(f"[Stems] Using {len(all_stems)} stems: {all_stems}")

    methods_cfg = [
        ("Our Method",   our_map),
        ("FOA-Attack",   foa_map),
        ("M-Attack",     m_map),
    ]

    all_per_image_rows: List[Dict] = []
    agg_rows: List[Dict] = []
    adv_dict_map: Dict[str, Dict] = {}

    for name, adv_map in methods_cfg:
        adv_dict_map[name] = adv_map
        per, agg = evaluate_method(
            name, clean_map, adv_map, all_stems, keywords, noise_dir)
        all_per_image_rows.extend(per)
        if agg:
            agg_rows.append(agg)
            print(f"[{name}] n={agg['n']:3d}  "
                  f"L1={agg['l1_mean']:.4f}  L2={agg['l2_mean']:.4f}  "
                  f"L∞={agg['linf_mean']:.1f}  PSNR={agg['psnr_mean']:.1f}dB")

    method_stem_pairs = [(name, stem, "")
                         for name, _ in methods_cfg
                         for stem in all_stems]
    noise_grid(method_stem_pairs, clean_map, adv_dict_map,
               os.path.join(args.out_dir, "noise_grid.png"))

    panel_dir = os.path.join(args.out_dir, "noise_panels")
    os.makedirs(panel_dir, exist_ok=True)
    for stem in all_stems:
        samples = []
        for name, adv_map in methods_cfg:
            if stem in clean_map and stem in adv_map:
                c = load_rgb(clean_map[stem])
                a = load_rgb(adv_map[stem])
                samples.append((name, c, a))
        if samples:
            save_noise_panel(samples,
                             os.path.join(panel_dir, f"stem_{stem}_noise_panel.png"))

    if keywords:
        print(f"\n[ASR] Keywords: {keywords}")
        print("[ASR] Note: Keyword matching requires VLM responses.")
        print("[ASR] Loading any saved detailed_report.json from our sample dirs...")

        our_responses: Dict[str, Dict] = {}
        if args.our_sample_root and os.path.isdir(args.our_sample_root):
            for d in sorted(os.listdir(args.our_sample_root)):
                rpt = os.path.join(args.our_sample_root, d, "detailed_report.json")
                if os.path.isfile(rpt):
                    try:
                        rpt_data = json.load(open(rpt))
                        idx = str(int(d.split("_")[1]))
                        our_responses[idx] = rpt_data.get("responses", {})
                    except Exception:
                        pass

        if our_responses:
            print(f"[ASR] Found VLM responses for {len(our_responses)} ours samples.")
            kw_rows = []
            for stem, resp in our_responses.items():
                orig_txt = resp.get("original", "")
                adv_txt  = resp.get("adversarial", "")
                row = {"stem": stem}
                for kw in keywords:
                    row[f"clean_{kw}"]   = keyword_hit(orig_txt, kw)
                    row[f"adv_{kw}"]     = keyword_hit(adv_txt, kw)
                    row[f"asr_{kw}"]     = (not row[f"clean_{kw}"]) and row[f"adv_{kw}"]
                kw_rows.append(row)

            kw_csv = os.path.join(args.out_dir, "keyword_matching_ours.csv")
            if kw_rows:
                with open(kw_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(kw_rows[0].keys()))
                    writer.writeheader(); writer.writerows(kw_rows)
                print(f"[ASR] Saved keyword CSV → {kw_csv}")

            asr_summary = {}
            for kw in keywords:
                hits = sum(r[f"asr_{kw}"] for r in kw_rows)
                asr_summary[kw] = hits / max(len(kw_rows), 1)
                print(f"  [Our Method] ASR({kw}) = {asr_summary[kw]:.2%} ({hits}/{len(kw_rows)})")

            for r in agg_rows:
                if r["method"] == "Our Method":
                    for kw in keywords:
                        r[f"asr_kw_{kw}"] = asr_summary.get(kw, 0.0)
        else:
            print("[ASR] No VLM responses found. To get keyword/ASR stats, run with "
                  "--our_sample_root pointing to the directory that contains sample_XXXXX_YY/ "
                  "subdirs with detailed_report.json files.")

        kw_labels = [f"(a) {keywords[0]}" if len(keywords) > 0 else "",
                     f"(b) {keywords[1]}" if len(keywords) > 1 else "",
                     f"(c) {keywords[2]}" if len(keywords) > 2 else ""]
        kw_labels = [k for k in kw_labels if k]
        print(f"\n  Keyword legend: " + "  ".join(kw_labels))

    per_csv = os.path.join(args.out_dir, "per_image_metrics.csv")
    if all_per_image_rows:
        with open(per_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_per_image_rows[0].keys()))
            writer.writeheader(); writer.writerows(all_per_image_rows)
        print(f"\n[Save] Per-image CSV → {per_csv}")

    agg_csv = os.path.join(args.out_dir, "aggregate_metrics.csv")
    if agg_rows:
        with open(agg_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader(); writer.writerows(agg_rows)
        print(f"[Save] Aggregate CSV → {agg_csv}")

    bar_chart(agg_rows,
              [("L1 Norm (↓ less visible)", "l1_mean"),
               ("L2 Norm (↓ less visible)", "l2_mean"),
               ("L∞ / max pixel (=16 expected)", "linf_mean"),
               ("PSNR dB (↑ less visible)", "psnr_mean")],
              "Perturbation Metrics: Our Method vs FOA-Attack vs M-Attack",
              os.path.join(args.out_dir, "perturbation_metrics.png"))
    print(f"[Save] Bar chart → {os.path.join(args.out_dir, 'perturbation_metrics.png')}")

    if keywords and any(f"asr_kw_{keywords[0]}" in r for r in agg_rows):
        asr_chart(agg_rows, keywords,
                  os.path.join(args.out_dir, "asr_chart.png"))
        print(f"[Save] ASR chart → {os.path.join(args.out_dir, 'asr_chart.png')}")

    print(f"\n[Done] All outputs in {args.out_dir}")

if __name__ == "__main__":
    main()
