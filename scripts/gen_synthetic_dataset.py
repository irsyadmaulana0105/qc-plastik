#!/usr/bin/env python3
"""Synthetic dataset generator: black specks on backlit, moving blue plastic film.

Produces YOLO-format pairs (synth_00001.jpg + synth_00001.txt) in --out, and optionally
a zip ready to upload to Google Drive for the Colab notebook.

What is simulated
    * blue translucent film: base color, backlight falloff, uneven illumination, longitudinal
      streaks along the web direction, glare bands, soft wrinkles, sensor grain
    * black_speck (class 0): irregular dark blobs, log-uniform size, varied opacity
    * motion blur along the web direction (applied to the specks, labels follow the blur)
    * hard negatives that must NOT be detected (unlabeled): bright dust, scratches, bubbles
      (disable with --no-hard-negatives); faint dark-blue smudges are opt-in via --smudges
    * clean frames (--neg-ratio) as pure negative samples

This is SYNTHETIC data: use it to test the pipeline and to pre-train. Always fine-tune and,
above all, validate on real photos from your own machine.

Usage:
    python gen_synthetic_dataset.py --n 300 --zip dataset_raw.zip
    python gen_synthetic_dataset.py --n 50 --axis y --max-blur 15 --min-size 2 --max-size 15
"""

import argparse
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------------- background
def lowfreq(rng, h, w, gh, gw):
    g = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def make_background(rng, w, h, axis):
    base = np.array([rng.uniform(150, 235), rng.uniform(70, 135), rng.uniform(5, 55)], np.float32)  # BGR
    img = np.empty((h, w, 3), np.float32)
    img[:] = base

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    illum = 1.0 + 0.04 * lowfreq(rng, h, w, 5, 8)           # uneven backlight
    illum *= 1.0 - rng.uniform(0.0, 0.12) * r2               # pad falloff towards the edges
    img *= illum[..., None]

    # longitudinal streaks: vary across the web, constant along it
    n = h if axis == "x" else w
    prof = rng.normal(0, 1, n).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), float(rng.uniform(1.5, 6))).ravel()
    prof *= rng.uniform(0.005, 0.02) / (prof.std() + 1e-6)
    streak = prof.reshape(-1, 1) if axis == "x" else prof.reshape(1, -1)
    img *= (1.0 + streak)[..., None]

    if rng.random() < 0.3:                                    # glare band
        ang = rng.uniform(0, math.pi)
        d = (xx - rng.uniform(0, w)) * math.cos(ang) + (yy - rng.uniform(0, h)) * math.sin(ang)
        band = np.exp(-(d / rng.uniform(60, 250)) ** 2)
        img += (band * rng.uniform(8, 28))[..., None] * np.array([1.0, 0.9, 0.7], np.float32)

    if rng.random() < 0.4:                                    # soft wrinkles
        layer = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(2, 6))):
            p0 = rng.uniform([0, 0], [w, h])
            ang, length = rng.uniform(0, math.pi), rng.uniform(300, 1200)
            pts = []
            for t in np.linspace(0, 1, 8):
                off = rng.normal(0, 15)
                pts.append([p0[0] + math.cos(ang) * length * t - math.sin(ang) * off,
                            p0[1] + math.sin(ang) * length * t + math.cos(ang) * off])
            cv2.polylines(layer, [np.array(pts, np.int32)], False,
                          float(rng.choice([-1.0, 1.0])), int(rng.integers(1, 3)))
        layer = cv2.GaussianBlur(layer, (0, 0), float(rng.uniform(2, 4)))
        img += (layer * rng.uniform(6, 16))[..., None]
    return img, base


# ----------------------------------------------------------------------------- blobs
def motion_kernel(length, axis):
    k = np.zeros((length, length), np.float32)
    if axis == "x":
        k[length // 2, :] = 1
    else:
        k[:, length // 2] = 1
    return k / k.sum()


def blob_alpha(rng, size, axis, blur_len, extra_soft=0.0):
    """Irregular blob alpha patch (float 0..1), motion-blurred along the web direction."""
    a = max(1.0, size / 2) * rng.uniform(0.7, 1.3)
    b = max(1.0, size / 2) * rng.uniform(0.5, 1.2)
    rot = rng.uniform(0, 2 * math.pi)
    n = int(rng.integers(8, 13))
    th = np.sort(rng.uniform(0, 2 * math.pi, n))
    rad = rng.uniform(0.7, 1.2, n)
    pad = int(max(a, b) * 1.5) + blur_len + int(extra_soft * 3) + 4
    s = 2 * pad + 1
    pts = []
    for t, r in zip(th, rad):
        x, y = a * r * math.cos(t), b * r * math.sin(t)
        pts.append([pad + x * math.cos(rot) - y * math.sin(rot),
                    pad + x * math.sin(rot) + y * math.cos(rot)])
    ss = 4                                                      # supersample for smooth tiny shapes
    m = np.zeros((s * ss, s * ss), np.uint8)
    cv2.fillPoly(m, [(np.array(pts) * ss).astype(np.int32)], 255)
    m = cv2.resize(m, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), float(rng.uniform(0.5, 1.0)) + extra_soft)
    if blur_len > 1:
        m = cv2.filter2D(m, -1, motion_kernel(blur_len, axis), borderType=cv2.BORDER_CONSTANT)
    return m, pad


def bbox_from_alpha(m):
    """Box around the visible part (>= 50% of the peak), in patch coordinates (x0, y0, x1, y1)."""
    ys, xs = np.where(m >= 0.5 * m.max())
    return xs.min() - 1, ys.min() - 1, xs.max() + 2, ys.max() + 2


def composite(img, m, cx, cy, pad, color, opacity):
    x0, y0 = cx - pad, cy - pad
    s = m.shape[0]
    a = (m * opacity)[..., None]
    region = img[y0:y0 + s, x0:x0 + s]
    img[y0:y0 + s, x0:x0 + s] = region * (1 - a) + np.asarray(color, np.float32) * a


def place(rng, w, h, pad, taken, min_dist=45):
    for _ in range(50):
        cx = int(rng.integers(pad + 3, w - pad - 3))
        cy = int(rng.integers(pad + 3, h - pad - 3))
        if all((cx - tx) ** 2 + (cy - ty) ** 2 > min_dist ** 2 for tx, ty in taken):
            return cx, cy
    return None


# ----------------------------------------------------------------------------- hard negatives
def add_hard_negatives(rng, img, base, axis, blur_len, smudges=False):
    h, w = img.shape[:2]
    if rng.random() < 0.5:                                      # bright dust
        for _ in range(int(rng.integers(3, 16))):
            m, pad = blob_alpha(rng, rng.uniform(2, 6), axis, blur_len)
            cx, cy = int(rng.integers(pad + 3, w - pad - 3)), int(rng.integers(pad + 3, h - pad - 3))
            composite(img, m, cx, cy, pad, (240, 230, 220), rng.uniform(0.3, 0.6))
    if rng.random() < 0.3:                                      # scratches
        layer = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(1, 4))):
            p = rng.uniform([0, 0], [w, h]); ang = rng.uniform(0, math.pi); length = rng.uniform(30, 200)
            q = p + length * np.array([math.cos(ang), math.sin(ang)])
            cv2.line(layer, tuple(map(int, p)), tuple(map(int, q)), 1.0, int(rng.integers(1, 3)))
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
        img += (layer * rng.uniform(20, 50))[..., None]
    if rng.random() < 0.25:                                     # air bubbles (bright rings)
        layer = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(1, 4))):
            c = rng.uniform([30, 30], [w - 30, h - 30]).astype(int)
            cv2.circle(layer, tuple(int(v) for v in c), int(rng.integers(4, 13)), 1.0, 1)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
        img += (layer * rng.uniform(15, 35))[..., None]
    if smudges and rng.random() < 0.25:                         # faint dark-blue smudge (opt-in, see --smudges)
        m, pad = blob_alpha(rng, rng.uniform(25, 70), axis, blur_len, extra_soft=3.0)
        cx, cy = int(rng.integers(pad + 3, w - pad - 3)), int(rng.integers(pad + 3, h - pad - 3))
        composite(img, m, cx, cy, pad, base * 0.55, rng.uniform(0.15, 0.30))


# ----------------------------------------------------------------------------- one image
def make_image(rng, args, positive):
    w, h = args.width, args.height
    img, base = make_background(rng, w, h, args.axis)
    blur_len = 1 + 2 * int(rng.integers(0, args.max_blur // 2 + 1)) if rng.random() > 0.3 else 1

    if not args.no_hard_negatives:
        add_hard_negatives(rng, img, base, args.axis, blur_len, args.smudges)

    boxes, taken = [], []
    if positive:
        n_specks = int(np.clip(1 + rng.poisson(1.5), 1, 8))
        for _ in range(n_specks):
            size = math.exp(rng.uniform(math.log(args.min_size), math.log(args.max_size)))
            m, pad = blob_alpha(rng, size, args.axis, blur_len)
            pos = place(rng, w, h, pad, taken)
            if pos is None:
                continue
            cx, cy = pos
            taken.append(pos)
            dark = (rng.uniform(0, 25), rng.uniform(0, 15), rng.uniform(0, 15))
            composite(img, m, cx, cy, pad, dark, rng.uniform(0.8, 1.0) if rng.random() > 0.15
                      else rng.uniform(0.55, 0.8))
            x0, y0, x1, y1 = bbox_from_alpha(m)
            boxes.append((cx - pad + x0, cy - pad + y0, cx - pad + x1, cy - pad + y1))

    img += rng.normal(0, rng.uniform(1.5, 3.5), img.shape).astype(np.float32)   # sensor grain
    return np.clip(img, 0, 255).astype(np.uint8), boxes


def yolo_lines(boxes, w, h):
    lines = []
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                     f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("dataset_synth"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--neg-ratio", type=float, default=0.3, help="fraction of clean frames")
    ap.add_argument("--min-size", type=float, default=3.0, help="smallest speck diameter in px")
    ap.add_argument("--max-size", type=float, default=22.0, help="largest speck diameter in px")
    ap.add_argument("--axis", choices=["x", "y"], default="x", help="direction the film moves in the image")
    ap.add_argument("--max-blur", type=int, default=9, help="max motion-blur length in px")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--smudges", action="store_true",
                    help="also add faint dark-blue smudges as unlabeled negatives (only if your QC "
                         "criteria say such marks are NOT defects)")
    ap.add_argument("--zip", type=Path, default=None, help="also write a zip (e.g. dataset_raw.zip)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    n_pos = n_boxes = 0
    names = []
    for i in range(1, args.n + 1):
        rng = np.random.default_rng(args.seed * 1_000_003 + i)
        positive = rng.random() >= args.neg_ratio
        img, boxes = make_image(rng, args, positive)
        stem = f"synth_{i:05d}"
        cv2.imwrite(str(args.out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        (args.out / f"{stem}.txt").write_text("\n".join(yolo_lines(boxes, args.width, args.height))
                                              + ("\n" if boxes else ""))
        names.append(stem)
        n_pos += bool(boxes)
        n_boxes += len(boxes)
        if i % 25 == 0 or i == args.n:
            print(f"{i}/{args.n} images | with specks: {n_pos} | boxes: {n_boxes}")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_STORED) as z:
            for stem in names:
                z.write(args.out / f"{stem}.jpg", f"dataset_raw/{stem}.jpg")
                z.write(args.out / f"{stem}.txt", f"dataset_raw/{stem}.txt")
        print(f"Zip: {args.zip} ({args.zip.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
