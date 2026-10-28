#!/usr/bin/env python3
"""Shrink loose YOLO boxes onto the actual dark speck inside them.

Why: hand-drawn boxes are often 5-10x larger than the speck itself (a 6 px dot inside a 60 px box).
The model then learns "a patch of plain plastic" instead of "a dark dot". This script looks inside
every box, finds the dark blob (blue channel: plastic is bright in blue, a black speck is not) and
writes a tight box. Originals are never modified.

Output (default folder: <src>_tight):
    images + tightened .txt files (same names)         -> use this folder with merge_real_data.py --real
    review/ *.jpg   side-by-side crops: LEFT = your box (yellow), RIGHT = new box (green)
    report.csv      per box: status (ok / unclear / big), old size, new size, contrast

Status:
    ok       dark blob found, box tightened
    unclear  nothing darker than the plastic inside the box (crease, faint mark?) -> box kept as is,
             CHECK these in review/ and re-draw or delete them with annotate.py
    big      the blob fills most of the box (large defect/smudge) -> box kept as is

Usage:
    python scripts/tighten_labels.py foto_asli
    python scripts/tighten_labels.py foto_asli --margin 3 --out foto_asli_tight
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_boxes(txt: Path, w: int, h: int):
    boxes = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                cx, cy, bw, bh = (float(x) for x in v[1:])
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def tighten_box(img, box, margin: int, pad: int = 24):
    """Return (new_box or None, status, contrast)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    rx1, ry1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    rx2, ry2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
    crop = img[ry1:ry2, rx1:rx2, 0]                       # blue channel (BGR index 0)
    if crop.shape[0] < 9 or crop.shape[1] < 9:
        return None, "unclear", 0.0
    k = 31 if min(crop.shape) >= 31 else (min(crop.shape) // 2) * 2 - 1
    bg = cv2.medianBlur(crop, max(k, 3)).astype(np.float32)
    diff = cv2.GaussianBlur(bg - crop.astype(np.float32), (0, 0), 1.0)
    sigma = 1.4826 * np.median(np.abs(diff - np.median(diff)))
    thr = max(10.0, 4.0 * sigma)

    # only look inside the user's box (+2 px tolerance)
    bx1, by1 = int(max(0, x1 - rx1 - 2)), int(max(0, y1 - ry1 - 2))
    bx2, by2 = int(min(crop.shape[1], x2 - rx1 + 2)), int(min(crop.shape[0], y2 - ry1 + 2))
    mask = np.zeros(diff.shape, np.uint8)
    mask[by1:by2, bx1:bx2] = (diff[by1:by2, bx1:bx2] > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_score = None, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        score = float(diff[lab == i].sum())
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None, "unclear", float(diff[by1:by2, bx1:bx2].max()) if by2 > by1 and bx2 > bx1 else 0.0

    sx, sy, sw, sh = (stats[best, c] for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    nb = (max(0, rx1 + sx - margin), max(0, ry1 + sy - margin),
          min(w, rx1 + sx + sw + margin), min(h, ry1 + sy + sh + margin))
    contrast = float(diff[lab == best].max())
    old_area = (x2 - x1) * (y2 - y1)
    status = "big" if (nb[2] - nb[0]) * (nb[3] - nb[1]) > 0.5 * old_area and old_area > 60 * 60 else "ok"
    return nb, status, contrast


def side_by_side(img, old, new, size=260):
    h, w = img.shape[:2]
    cx, cy = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
    half = max(old[2] - old[0], old[3] - old[1]) / 2 + 30
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    panel = []
    for b, col in ((old, (0, 255, 255)), (new, (0, 255, 0))):
        c = img[y0:y1, x0:x1].copy()
        if b is not None:
            cv2.rectangle(c, (int(b[0] - x0), int(b[1] - y0)), (int(b[2] - x0), int(b[3] - y0)), col, 1)
        panel.append(cv2.resize(c, (size, size), interpolation=cv2.INTER_NEAREST))
    return np.hstack(panel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="folder with images + YOLO .txt")
    ap.add_argument("--out", type=Path, default=None, help="default: <src>_tight")
    ap.add_argument("--margin", type=int, default=3, help="pixels added around the blob (default 3)")
    args = ap.parse_args()

    out = args.out or args.src.with_name(args.src.name + "_tight")
    (out / "review").mkdir(parents=True, exist_ok=True)
    rows, counts = [], {"ok": 0, "unclear": 0, "big": 0}
    for img_path in sorted(p for p in args.src.iterdir() if p.suffix.lower() in IMG_EXTS):
        txt = img_path.with_suffix(".txt")
        shutil.copy2(img_path, out / img_path.name)
        if not txt.exists():
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lines = []
        for bi, box in enumerate(read_boxes(txt, w, h)):
            nb, status, contrast = tighten_box(img, box, args.margin)
            counts[status] += 1
            final = nb if (nb is not None and status == "ok") else box
            cx, cy = (final[0] + final[2]) / 2 / w, (final[1] + final[3]) / 2 / h
            bw, bh = (final[2] - final[0]) / w, (final[3] - final[1]) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            rows.append([img_path.name, bi, status, f"{box[2]-box[0]:.0f}x{box[3]-box[1]:.0f}",
                         f"{final[2]-final[0]:.0f}x{final[3]-final[1]:.0f}", f"{contrast:.0f}"])
            cv2.imwrite(str(out / "review" / f"{status}_{img_path.stem}_{bi}.jpg"),
                        side_by_side(img, box, nb))
        (out / txt.name).write_text("\n".join(lines) + ("\n" if lines else ""))

    with open(out / "report.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "box", "status", "old_wxh_px", "new_wxh_px", "contrast"])
        wr.writerows(rows)
    print(f"Boxes: ok {counts['ok']} | unclear {counts['unclear']} | big {counts['big']}")
    print(f"Output: {out}   (cek review/ terutama yang berawalan unclear_ dan big_)")


if __name__ == "__main__":
    main()
