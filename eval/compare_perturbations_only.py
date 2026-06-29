import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".JPEG"}

def _load_rgb(path: str, size: Optional[int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.float32)

@dataclass
class Pair:
    sample_id: str
    clean_path: str
    adv_path: str

def _parse_stems(s: Optional[str]) -> Optional[Set[str]]:
    if not s or not str(s).strip():
        return None
    return {x.strip() for x in str(s).split(",") if x.strip()}

def _infer_clean_adv_pairs(
    clean_dir: str,
    adv_dir: str,
    allowed_stems: Optional[Set[str]] = None,
    shallow: bool = False,
) -> List[Pair]:
    adv_paths: List[str] = []
    if shallow:
        if not os.path.isdir(adv_dir):
            raise RuntimeError(f"adv_dir is not a directory: {adv_dir}")
        for fn in sorted(os.listdir(adv_dir)):
            p = os.path.join(adv_dir, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1] in VALID_EXT:
                adv_paths.append(p)
    else:
        for dp, _, fns in os.walk(adv_dir):
            for fn in fns:
                if os.path.splitext(fn)[1] in VALID_EXT:
                    adv_paths.append(os.path.join(dp, fn))
        adv_paths.sort()

    pairs: List[Pair] = []
    for adv_p in adv_paths:
        rel = os.path.relpath(adv_p, adv_dir)
        sub = os.path.dirname(rel)
        stem = os.path.splitext(os.path.basename(rel))[0]
        if allowed_stems is not None and stem not in allowed_stems:
            continue

        clean_p = None
        for ext in VALID_EXT:
            cand = os.path.join(clean_dir, sub, stem + ext)
            if os.path.isfile(cand):
                clean_p = cand
                break
        if clean_p is None:
            for dp, _, fns in os.walk(clean_dir):
                for fn in fns:
                    if os.path.splitext(fn)[0] == stem and os.path.splitext(fn)[1] in VALID_EXT:
                        clean_p = os.path.join(dp, fn)
                        break
                if clean_p is not None:
                    break
        if clean_p is None:
            continue

        sample_id = os.path.join(sub, stem).replace(os.sep, "/")
        pairs.append(Pair(sample_id=sample_id, clean_path=clean_p, adv_path=adv_p))
    return pairs

def _psnr(clean01: np.ndarray, adv01: np.ndarray) -> float:
    mse = float(np.mean((clean01 - adv01) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))

def _delta_kl_to_uniform(delta_255: np.ndarray, eps: float = 1e-12) -> float:
    d = np.rint(delta_255).astype(np.int32)
    d = np.clip(d, -16, 16)
    idx = d + 16
    hist = np.bincount(idx.reshape(-1), minlength=33).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    q = np.ones_like(p) / float(p.size)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

def evaluate_method(
    name: str,
    clean_dir: str,
    adv_dir: str,
    input_res: Optional[int],
    allowed_stems: Optional[Set[str]] = None,
    shallow: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    pairs = _infer_clean_adv_pairs(
        clean_dir, adv_dir, allowed_stems=allowed_stems, shallow=shallow
    )
    if not pairs:
        raise RuntimeError(f"[{name}] No clean/adv pairs matched. clean_dir='{clean_dir}', adv_dir='{adv_dir}'")

    l1_list: List[float] = []
    l2_list: List[float] = []
    linf_list: List[float] = []
    psnr_list: List[float] = []
    kl_list: List[float] = []
    per_image: List[Dict[str, float]] = []

    for p in pairs:
        clean = _load_rgb(p.clean_path, input_res)
        adv = _load_rgb(p.adv_path, input_res)
        if clean.shape != adv.shape:
            h = min(clean.shape[0], adv.shape[0])
            w = min(clean.shape[1], adv.shape[1])
            clean = clean[:h, :w]
            adv = adv[:h, :w]

        delta = adv - clean
        delta01 = delta / 255.0

        numel = float(delta01.size)
        l1 = float(np.sum(np.abs(delta01)) / max(numel, 1.0))
        l2 = float(np.sqrt(np.sum(delta01 ** 2)) / max(np.sqrt(numel), 1.0))
        linf = float(np.max(np.abs(delta)))
        ps = _psnr(clean / 255.0, adv / 255.0)
        kl = _delta_kl_to_uniform(delta)

        l1_list.append(l1)
        l2_list.append(l2)
        linf_list.append(linf)
        psnr_list.append(ps)
        kl_list.append(kl)
        per_image.append(
            {
                "method": name,
                "sample_id": p.sample_id,
                "l1": l1,
                "l2": l2,
                "linf_255": linf,
                "psnr_db": ps,
                "delta_kl_uniform": kl,
            }
        )

    agg = {
        "method": name,
        "n": float(len(pairs)),
        "l1_mean": float(np.mean(l1_list)),
        "l2_mean": float(np.mean(l2_list)),
        "linf_mean_255": float(np.mean(linf_list)),
        "psnr_mean_db": float(np.mean(psnr_list)),
        "delta_kl_uniform_mean": float(np.mean(kl_list)),
    }
    return agg, per_image

def main():
    ap = argparse.ArgumentParser(description="Compare attacks using imperceptibility + perturbation statistics only.")
    ap.add_argument("--clean_dir", required=True, help="Clean/source images root.")
    ap.add_argument("--input_res", type=int, default=224, help="Resize images to this size before metrics.")
    ap.add_argument("--out_csv", default="perturbation_comparison.csv", help="Output aggregate summary CSV path.")
    ap.add_argument(
        "--out_per_image_csv",
        default=None,
        help="Optional path for per-sample_id rows (all methods).",
    )

    ap.add_argument("--yours_adv_dir", default=None, help="Your method: adversarial images directory.")
    ap.add_argument("--yours_name", default="Yours", help="Label for your method.")
    ap.add_argument("--foa_adv_dir", default=None, help="FOA-Attack adversarial images directory.")
    ap.add_argument("--foa_name", default="FOA-Attack", help="Label for FOA method.")
    ap.add_argument("--m_adv_dir", default=None, help="M-Attack adversarial images directory.")
    ap.add_argument("--m_name", default="M-Attack", help="Label for M method.")
    ap.add_argument(
        "--stems",
        default=None,
        help="Comma-separated stems to include only (e.g. 0,1,2,...,10 for first 11 ids).",
    )
    ap.add_argument(
        "--m_shallow",
        action="store_true",
        help="For M-Attack: only use PNG/JPG files directly in --m_adv_dir (no subfolders). "
        "Use this when adv_dir also contains adv_images/ to avoid duplicate stems.",
    )
    args = ap.parse_args()

    allowed = _parse_stems(args.stems)

    runs = []
    if args.yours_adv_dir:
        runs.append((args.yours_name, args.yours_adv_dir, False))
    if args.foa_adv_dir:
        runs.append((args.foa_name, args.foa_adv_dir, False))
    if args.m_adv_dir:
        runs.append((args.m_name, args.m_adv_dir, bool(args.m_shallow)))
    if not runs:
        ap.error("Provide at least one of --yours_adv_dir, --foa_adv_dir, --m_adv_dir")

    out_agg: List[Dict[str, float]] = []
    all_per: List[Dict[str, float]] = []
    for name, adv_dir, shallow in runs:
        agg, per = evaluate_method(
            name,
            args.clean_dir,
            adv_dir,
            args.input_res,
            allowed_stems=allowed,
            shallow=shallow,
        )
        out_agg.append(agg)
        all_per.extend(per)

    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_agg[0].keys()))
        w.writeheader()
        for r in out_agg:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    if args.out_per_image_csv:
        pcsv = args.out_per_image_csv
        os.makedirs(os.path.dirname(pcsv) or ".", exist_ok=True)
        fields = ["method", "sample_id", "l1", "l2", "linf_255", "psnr_db", "delta_kl_uniform"]
        with open(pcsv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_per:
                w.writerow({k: (f"{r[k]:.6f}" if k != "method" and k != "sample_id" else r[k]) for k in fields})

    cols = ["method", "n", "l1_mean", "l2_mean", "linf_mean_255", "psnr_mean_db", "delta_kl_uniform_mean"]
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)
    widths = {c: max(len(c), max(len(fmt(r[c])) for r in out_agg)) for c in cols}
    def line(r):
        return " | ".join(fmt(r[c]).ljust(widths[c]) for c in cols)

    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in out_agg:
        print(line(r))
    print(f"\nWrote: {out_csv}")
    if args.out_per_image_csv:
        print(f"Wrote: {args.out_per_image_csv}")

if __name__ == "__main__":
    main()

