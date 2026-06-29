
import argparse
import os
import sys
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family":       "STIXGeneral",
    "font.size":         9,
    "font.weight":       "bold",
    "axes.titleweight":  "bold",
    "axes.labelweight":  "bold",
    "mathtext.fontset":  "stix",
})

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(_BASE, "resources/images/bigscale_copy/nips17")
TARGET_DIR = os.path.join(_BASE, "resources/images/target_images_copy/1")

_BASELINE = os.environ.get("BASELINE_DIR", "")

ATTACKS = [
    (
        "M-Attack",
        os.path.join(_BASELINE, "foa_attack_output"),
    ),
    (
        "FOA-Attack",
        os.path.join(_BASELINE, "foa_attack_output3/adv_images/cluster_3/nips17"),
    ),
    (
        "MIRROR",
        os.environ.get("MIRROR_ADV_DIR", os.path.join(_BASE, "mirror_attack_output", "img", "nips17")),
    ),
]

DEFAULT_INDICES = [12, 16, 24]
AMPLIFY        = 20
OUT_PATH       = os.path.join(_BASE, "eval/figure3_perturbations.png")
DPI            = 300
TITLE_FONTSIZE = 11
PANEL_GAP      = 0.04

def _find_image(directory: str, stem: int) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = os.path.join(directory, f"{stem}{ext}")
        if os.path.isfile(p):
            return p
    return None

def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_to(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, (hw[1], hw[0]),
                      interpolation=cv2.INTER_LANCZOS4)

def amplified_noise(src: np.ndarray, adv: np.ndarray,
                    amplify: int = AMPLIFY) -> np.ndarray:
    delta = adv.astype(np.float32) - src.astype(np.float32)
    vis   = np.clip(delta * amplify + 128, 0, 255)
    return vis.astype(np.uint8)

def compute_metrics(src: np.ndarray, adv: np.ndarray) -> dict:
    d    = adv.astype(np.float32) - src.astype(np.float32)
    mse  = float(np.mean(d ** 2))
    return {
        "l1":   float(np.mean(np.abs(d))) / 255.0,
        "l2":   float(np.sqrt(mse)) / 255.0,
        "psnr": 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf"),
        "linf": float(np.max(np.abs(d))),
    }

def load_row(idx: int, attacks: list[tuple[str, str]], cell_hw: tuple[int, int]):
    src_path = _find_image(SOURCE_DIR, idx)
    tgt_path = _find_image(TARGET_DIR, idx)

    if src_path is None:
        raise FileNotFoundError(f"Source image {idx} not found in {SOURCE_DIR}")
    if tgt_path is None:
        print(f"[WARN] Target image {idx} not found in {TARGET_DIR} — showing blank.")

    src = resize_to(read_rgb(src_path), cell_hw)
    tgt = resize_to(read_rgb(tgt_path), cell_hw) if tgt_path else np.full((*cell_hw, 3), 200, np.uint8)

    attack_data = []
    for label, adv_dir in attacks:
        adv_path = _find_image(adv_dir, idx)
        if adv_path is None:
            print(f"[WARN] Adversarial {idx} not found in {adv_dir}")
            blank = np.full((*cell_hw, 3), 200, np.uint8)
            attack_data.append((label, blank, blank, None))
        else:
            adv_full = read_rgb(adv_path)
            src_full = resize_to(read_rgb(src_path), (adv_full.shape[0], adv_full.shape[1]))
            m        = compute_metrics(src_full, adv_full)
            adv      = resize_to(adv_full, cell_hw)
            noise    = amplified_noise(src, adv)
            attack_data.append((label, adv, noise, m))

    return src, tgt, attack_data

def make_figure(indices: list[int], attacks: list[tuple[str, str]],
                amplify: int = AMPLIFY,
                out_path: str = OUT_PATH, dpi: int = DPI) -> None:

    n_rows   = len(indices)
    n_attack = len(attacks)
    n_cols   = 1 + n_attack

    cell_hw = (224, 224)

    rows_data = []
    for idx in indices:
        src, tgt, attack_data = load_row(idx, attacks, cell_hw)
        rows_data.append((src, tgt, attack_data))

    METRIC_FS = 9.0

    import matplotlib.gridspec as gridspec

    total_cols  = n_cols * 2 + 1
    spacer_idx  = n_cols

    fig = plt.figure(figsize=(n_cols * 2.2 * 2 + 0.5, n_rows * 3.0 + 0.6))

    col_widths = [1] * n_cols + [PANEL_GAP * n_cols] + [1] * n_cols
    gs = gridspec.GridSpec(
        n_rows, total_cols,
        figure=fig,
        wspace=0.02, hspace=0.04,
        width_ratios=col_widths,
    )

    attack_labels = [lbl for lbl, _ in attacks]
    left_labels   = ["Source"] + attack_labels
    right_labels  = ["Target"] + [f"{lbl}\nnoise" for lbl in attack_labels]

    def _metric_text(m: Optional[dict]) -> str:
        if m is None:
            return ""
        return f"L1={m['l1']:.4f}   L2={m['l2']:.4f}\nPSNR={m['psnr']:.2f} dB"

    for row_i, (src, tgt, attack_data) in enumerate(rows_data):
        left_imgs    = [src]  + [ad for _, ad, _, _ in attack_data]
        right_imgs   = [tgt]  + [ns for _, _, ns, _ in attack_data]
        left_metrics  = [None] + [m  for _, _, _, m  in attack_data]
        right_metrics = [None] + [m  for _, _, _, m  in attack_data]

        for col_i in range(n_cols):
            ax_l = fig.add_subplot(gs[row_i, col_i])
            ax_l.imshow(left_imgs[col_i])
            ax_l.set_xticks([]); ax_l.set_yticks([])
            for spine in ax_l.spines.values():
                spine.set_visible(False)
            if row_i == 0:
                ax_l.set_title(left_labels[col_i],
                               fontsize=TITLE_FONTSIZE, fontweight="bold",
                               pad=3)
            if col_i > 0 and left_metrics[col_i] is not None:
                ax_l.set_xlabel(_metric_text(left_metrics[col_i]),
                                fontsize=METRIC_FS, labelpad=4,
                                color="#111111", ha="center")

            ax_r = fig.add_subplot(gs[row_i, spacer_idx + 1 + col_i])
            ax_r.imshow(right_imgs[col_i])
            ax_r.set_xticks([]); ax_r.set_yticks([])
            for spine in ax_r.spines.values():
                spine.set_visible(False)
            if row_i == 0:
                ax_r.set_title(right_labels[col_i],
                               fontsize=TITLE_FONTSIZE, fontweight="bold",
                               pad=3)

        ax_sp = fig.add_subplot(gs[row_i, spacer_idx])
        ax_sp.axis("off")

    fig.suptitle(
        f"Figure 3.  Visualization of adversarial images and perturbations "
        f"(perturbation amplified ×{amplify})",
        fontsize=9, y=0.01, va="bottom",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Figure-3 perturbation visualization for adversarial attack papers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--indices", nargs="+", type=int, default=DEFAULT_INDICES,
                   help="Image indices (filename stems) to include as rows.")
    p.add_argument("--source_dir", default=SOURCE_DIR)
    p.add_argument("--target_dir", default=TARGET_DIR)
    p.add_argument("--amplify", type=int, default=AMPLIFY,
                   help="Perturbation amplification factor (10–30 for papers).")
    p.add_argument("--out", default=OUT_PATH,
                   help="Output PNG path.")
    p.add_argument("--dpi", type=int, default=DPI)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    SOURCE_DIR = args.source_dir
    TARGET_DIR = args.target_dir
    make_figure(
        indices   = args.indices,
        attacks   = ATTACKS,
        amplify   = args.amplify,
        out_path  = args.out,
        dpi       = args.dpi,
    )
