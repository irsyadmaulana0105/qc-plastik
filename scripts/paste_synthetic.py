#!/usr/bin/env python3
"""Realistic synthetic data: paste YOUR real specks onto YOUR real plastic photos.

How it works
    1. Every labelled box in --src is searched for the dark blob inside it (blue channel).
       Blobs that are clearly dark become a "speck library": a tiny patch + soft mask + how much
       it darkens each colour channel (a ratio, so it also looks right on a brighter/darker background).
    2. Background pool = your clean photos + photos whose labelled specks could all be located
       (those specks are inpainted away). Photos with doubtful boxes (crease/faint mark with no clear dot)
       are NOT used as backgrounds, so unlabelled "maybe specks" never become false negatives.
    3. Each output image = a background (random flip, small brightness change) with 1..N library
       specks pasted at random places on the PLASTIC only (tile floor / plastic edge are avoided),
       randomly rotated / scaled / contrast-adjusted. The YOLO label is computed from the pasted mask.

Output (default folder: synth_paste):  synth_paste_00001.jpg/.txt ...   (+ zip with --zip)
The prefix `synth_` keeps them train-only in the Colab notebook (validation stays real photos).
The zip contains a top-level dataset_raw/ folder, so it can be the --base-zip of merge_real_data.py.

Usage:
    python scripts/paste_synthetic.py --src foto_asli_tight --n 300 --zip
    python scripts/paste_synthetic.py --src foto_asli_tight --n 300 --old-zip dataset_raw_sintetis.zip --old-n 100 --zip
    python scripts/merge_real_data.py --real foto_asli_tight --base-zip synth_paste.zip --out dataset_raw_merged.zip
"""

import argparse
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------- reading
def read_boxes(txt: Path, w: int, h: int):
    boxes = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                cx, cy, bw, bh = (float(x) for x in v[1:])
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def find_blob(img, box, pad=24):
    """Dark blob inside `box` (blue channel). Returns dict or None."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    rx1, ry1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    rx2, ry2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
    crop = img[ry1:ry2, rx1:rx2, 0]
    if min(crop.shape) < 15:
        return None
    k = 31 if min(crop.shape) >= 31 else (min(crop.shape) // 2) * 2 - 1
    bg = cv2.medianBlur(crop, k).astype(np.float32)
    diff = cv2.GaussianBlur(bg - crop.astype(np.float32), (0, 0), 1.0)
    sigma = 1.4826 * np.median(np.abs(diff - np.median(diff)))
    thr = max(10.0, 4.0 * sigma)
    bx1, by1 = int(max(0, x1 - rx1 - 2)), int(max(0, y1 - ry1 - 2))
    bx2, by2 = int(min(crop.shape[1], x2 - rx1 + 2)), int(min(crop.shape[0], y2 - ry1 + 2))
    mask = np.zeros(diff.shape, np.uint8)
    mask[by1:by2, bx1:bx2] = (diff[by1:by2, bx1:bx2] > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_score = None, 0.0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        s = float(diff[lab == i].sum())
        if s > best_score:
            best, best_score = i, s
    if best is None:
        return None
    return dict(mask=(lab == best).astype(np.uint8), ox=rx1, oy=ry1, contrast=float(diff[lab == best].max()),
                bbox=(rx1 + stats[best, 0], ry1 + stats[best, 1], stats[best, 2], stats[best, 3]))


def make_patch(img, blob, ring=7):
    """Cut a speck patch: ratio image (pixel / local background), soft alpha."""
    h, w = img.shape[:2]
    bx, by, bw, bh = blob["bbox"]
    x0, y0, x1, y1 = max(0, bx - ring), max(0, by - ring), min(w, bx + bw + ring), min(h, by + bh + ring)
    # local background colour from a larger median-blurred crop
    X0, Y0, X1, Y1 = max(0, x0 - 20), max(0, y0 - 20), min(w, x1 + 20), min(h, y1 + 20)
    big = img[Y0:Y1, X0:X1]
    k = 31 if min(big.shape[:2]) >= 31 else (min(big.shape[:2]) // 2) * 2 - 1
    bg = cv2.medianBlur(big, max(k, 3)).astype(np.float32)[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    pix = img[y0:y1, x0:x1].astype(np.float32)
    ratio = np.clip(pix / np.maximum(bg, 1.0), 0.0, 1.0)

    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    full = np.zeros(img.shape[:2], np.uint8)
    mm = blob["mask"]
    full[blob["oy"]:blob["oy"] + mm.shape[0], blob["ox"]:blob["ox"] + mm.shape[1]] = mm
    m = full[y0:y1, x0:x1]
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.9)
    alpha = np.clip(alpha * 1.4, 0, 1)
    return ratio, alpha


# ------------------------------------------------------------------------ backgrounds
def plastic_mask(img, margin=28):
    """True where it is blue plastic (not tile floor, not the plastic edge)."""
    b = cv2.GaussianBlur(img[:, :, 0].astype(np.float32) - img[:, :, 2].astype(np.float32), (0, 0), 3)
    m = (b > 60).astype(np.uint8)
    m = cv2.erode(m, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return m


def inpaint_blobs(img, blobs):
    if not blobs:
        return img
    mask = np.zeros(img.shape[:2], np.uint8)
    for b in blobs:
        mm = b["mask"]
        mask[b["oy"]:b["oy"] + mm.shape[0], b["ox"]:b["ox"] + mm.shape[1]] |= mm
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    return cv2.inpaint(img, mask, 4, cv2.INPAINT_TELEA)


# --------------------------------------------------------------------------- pasting
def augment_patch(ratio, alpha, rng):
    ang = rng.uniform(0, 360)
    sc = rng.uniform(0.75, 1.6)
    gain = rng.uniform(0.8, 1.4)
    ph, pw = alpha.shape
    d = int(np.ceil(np.hypot(ph, pw) * sc)) + 4
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), ang, sc)
    M[0, 2] += d / 2 - pw / 2
    M[1, 2] += d / 2 - ph / 2
    r = cv2.warpAffine(ratio, M, (d, d), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    a = cv2.warpAffine(alpha, M, (d, d), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if rng.random() < 0.3:                               # slight softness / motion-ish blur
        s = rng.uniform(0.4, 1.0)
        r = cv2.GaussianBlur(r, (0, 0), s)
        a = cv2.GaussianBlur(a, (0, 0), s)
    r = np.clip(1.0 - gain * (1.0 - r), 0.0, 1.0)
    return r, np.clip(a, 0, 1)


def paste(target, r, a, cx, cy, margin_px=3):
    """Paste in place. Returns tight box (x1,y1,x2,y2) or None."""
    h, w = target.shape[:2]
    d = a.shape[0]
    x0, y0 = int(cx - d / 2), int(cy - d / 2)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(d, w - x0), min(d, h - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    ys, xs = np.where(a > 0.25)
    if len(xs) == 0:
        return None
    if xs.min() + x0 < 0 or ys.min() + y0 < 0 or xs.max() + x0 >= w or ys.max() + y0 >= h:
        return None                                      # speck would be cut by the image border
    reg = target[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1].astype(np.float32)
    aa = a[sy0:sy1, sx0:sx1, None]
    rr = r[sy0:sy1, sx0:sx1]
    out = reg * (1.0 - aa * (1.0 - rr))
    target[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1] = np.clip(out, 0, 255).astype(np.uint8)
    return (max(0, xs.min() + x0 - margin_px), max(0, ys.min() + y0 - margin_px),
            min(w, xs.max() + x0 + 1 + margin_px), min(h, ys.max() + y0 + 1 + margin_px))


def overlaps(box, others, gap=25):
    for o in others:
        if box[0] < o[2] + gap and box[2] > o[0] - gap and box[1] < o[3] + gap and box[3] > o[1] - gap:
            return True
    return False


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder with real images + YOLO .txt (e.g. foto_asli_tight)")
    ap.add_argument("--out", type=Path, default=Path("synth_paste"))
    ap.add_argument("--n", type=int, default=300, help="number of images to generate")
    ap.add_argument("--neg-ratio", type=float, default=0.2, help="share of images with no pasted speck")
    ap.add_argument("--max-specks", type=int, default=4)
    ap.add_argument("--min-contrast", type=float, default=14.0, help="only use library blobs at least this dark (blue channel, 0-255)")
    ap.add_argument("--max-blob", type=int, default=30, help="ignore blobs larger than this (px); bigger ones are usually creases/streaks")
    ap.add_argument("--max-aspect", type=float, default=3.0, help="ignore elongated blobs (long side / short side above this)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip with a top-level dataset_raw/ folder")
    ap.add_argument("--old-zip", type=Path, default=None, help="optional: older synthetic zip to sample from")
    ap.add_argument("--old-n", type=int, default=0, help="how many images to copy from --old-zip")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    library, backgrounds = [], []
    n_img = n_box = n_doubt = 0
    for p in sorted(x for x in args.src.iterdir() if x.suffix.lower() in IMG_EXTS):
        txt = p.with_suffix(".txt")
        if not txt.exists():
            continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        boxes = read_boxes(txt, w, h)
        n_img += 1
        good, doubtful = [], 0
        for bx in boxes:
            n_box += 1
            blob = find_blob(img, bx)
            bw, bh = (blob["bbox"][2], blob["bbox"][3]) if blob else (0, 0)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if (blob and blob["contrast"] >= args.min_contrast and max(bw, bh) <= args.max_blob
                    and aspect <= args.max_aspect):
                good.append(blob)
                library.append(make_patch(img, blob))
            else:
                doubtful += 1
        n_doubt += doubtful
        if doubtful == 0:                                # clean photo, or every box located -> safe background
            bg = inpaint_blobs(img, good)
            backgrounds.append((bg, plastic_mask(bg)))
    if not library or not backgrounds:
        raise SystemExit("Not enough usable specks/backgrounds found. Check --src (needs images + .txt).")
    print(f"Source: {n_img} images, {n_box} boxes -> speck library {len(library)}, "
          f"doubtful boxes skipped {n_doubt}, background photos {len(backgrounds)}")

    args.out.mkdir(parents=True, exist_ok=True)
    made, total_boxes = 0, 0
    names = []
    tries = 0
    while made < args.n and tries < args.n * 20:
        tries += 1
        bg, pm = backgrounds[rng.integers(len(backgrounds))]
        img = bg.copy()
        pmask = pm
        if rng.random() < 0.5:
            img, pmask = img[:, ::-1].copy(), pmask[:, ::-1]
        if rng.random() < 0.5:
            img, pmask = img[::-1].copy(), pmask[::-1]
        gain = rng.uniform(0.9, 1.1)
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        boxes = []
        k = 0 if rng.random() < args.neg_ratio else int(rng.integers(1, args.max_specks + 1))
        ys, xs = np.where(pmask > 0)
        if k and len(xs) == 0:
            continue
        for _ in range(k):
            for _try in range(30):
                r, a = augment_patch(*library[rng.integers(len(library))], rng)
                i = rng.integers(len(xs))
                cx, cy = int(xs[i]), int(ys[i])
                d = a.shape[0]
                probe = (cx - d // 2, cy - d // 2, cx + d // 2, cy + d // 2)
                if overlaps(probe, boxes, gap=10):
                    continue
                b = paste(img, r, a, cx, cy)
                if b is not None:
                    boxes.append(b)
                    break
        if k and not boxes:
            continue
        h, w = img.shape[:2]
        stem = f"synth_paste_{made + 1:05d}"
        cv2.imwrite(str(args.out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        lines = [f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}"
                 for b in boxes]
        (args.out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        names.append(stem)
        made += 1
        total_boxes += len(boxes)
    print(f"Generated {made} images ({total_boxes} speck boxes, {made - sum(1 for s in names if (args.out / (s + '.txt')).stat().st_size)} clean) in {args.out}")

    if args.zip:
        zp = args.out.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for s in names:
                z.write(args.out / f"{s}.jpg", f"dataset_raw/{s}.jpg")
                z.write(args.out / f"{s}.txt", f"dataset_raw/{s}.txt")
            if args.old_zip and args.old_n > 0:
                with zipfile.ZipFile(args.old_zip) as oz:
                    stems = sorted({Path(n).stem for n in oz.namelist() if n.lower().endswith(".jpg")})
                    random.shuffle(stems)
                    taken = 0
                    for s in stems:
                        if taken >= args.old_n:
                            break
                        jn = next((n for n in oz.namelist() if Path(n).stem == s and n.lower().endswith(".jpg")), None)
                        tn = next((n for n in oz.namelist() if Path(n).stem == s and n.lower().endswith(".txt")), None)
                        if jn and tn:
                            z.writestr(f"dataset_raw/{Path(jn).name}", oz.read(jn))
                            z.writestr(f"dataset_raw/{Path(tn).name}", oz.read(tn))
                            taken += 1
                    print(f"Copied {taken} images from old synthetic zip")
        print(f"Zip: {zp}  ({zp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
