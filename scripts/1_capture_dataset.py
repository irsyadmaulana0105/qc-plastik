#!/usr/bin/env python3
"""Dataset capture tool for backlit blue plastic film (roll / per-meter material).

Saved frames are ALWAYS full-resolution color JPEGs. The channel views below only
change the on-screen preview, so you can judge which channel gives the best speck contrast.

Keys:
    s      - save current frame to dataset_raw/
    a      - toggle auto-capture (every --interval seconds)
    0/1/2/3- preview: 0 = color, 1 = red, 2 = green, 3 = blue channel
    q      - quit

Examples:
    python 1_capture_dataset.py
    python 1_capture_dataset.py --interval 3 --exposure -7
    python 1_capture_dataset.py --camera 1 --interval 2 --min-sharpness 60

Notes:
    * --exposure uses OpenCV's CAP_PROP_EXPOSURE. Its scale is backend/driver dependent
      (DirectShow on Windows: usually negative log2 seconds, e.g. -7 ~ 1/128 s; V4L2 on Linux:
      often units of 100 us). Try a few values and watch the image.
    * --min-sharpness skips blurred frames in auto-capture (useful while the roll is moving).
      The sharpness value is shown on screen; read it on a still, then set a threshold below it.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_FPS = 30
JPEG_QUALITY = 98          # near-lossless, keeps small specks intact
PREVIEW_WIDTH = 1280       # display only; saved frames stay full resolution
WINDOW_NAME = "Dataset Capture  [s] save  [a] auto  [0-3] view  [q] quit"
VIEW_NAMES = {0: "COLOR", 1: "RED", 2: "GREEN", 3: "BLUE"}


def open_camera(index: int, exposure: float | None) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")

    # MJPG is required by most USB cameras to reach 1080p at 30 FPS.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if exposure is not None:
        # 0.25 = manual mode on V4L2/DirectShow backends, 0.75 = auto (OpenCV convention)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print(f"Manual exposure requested: {exposure} (read back: {cap.get(cv2.CAP_PROP_EXPOSURE)})")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened at {w}x{h}")
    if (w, h) != (FRAME_WIDTH, FRAME_HEIGHT):
        print(f"WARNING: requested {FRAME_WIDTH}x{FRAME_HEIGHT}, camera returned {w}x{h}")
    return cap


def channel_view(frame: np.ndarray, mode: int) -> np.ndarray:
    """Return a BGR image for preview. OpenCV frames are BGR: B=0, G=1, R=2."""
    if mode == 0:
        return frame
    idx = {1: 2, 2: 1, 3: 0}[mode]
    return cv2.cvtColor(frame[:, :, idx], cv2.COLOR_GRAY2BGR)


def channel_means(frame: np.ndarray) -> tuple[float, float, float]:
    """Mean brightness of (R, G, B) on a downscaled copy (cheap)."""
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    b, g, r = (float(small[:, :, i].mean()) for i in range(3))
    return r, g, b


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian on a downscaled gray copy: higher = sharper."""
    gray = cv2.cvtColor(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def put(img, text, y, color=(0, 255, 0)):
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("dataset_raw"))
    ap.add_argument("--interval", type=float, default=2.0, help="auto-capture period in seconds")
    ap.add_argument("--exposure", type=float, default=None,
                    help="manual exposure value (driver dependent); omit to keep auto exposure")
    ap.add_argument("--min-sharpness", type=float, default=0.0,
                    help="auto-capture skips frames below this sharpness (0 = disabled)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    saved = len(list(args.out.glob("*.jpg")))
    view_mode = 0
    auto = False
    next_auto = 0.0
    flash_until = 0.0
    skipped_blur = 0

    def save(frame: np.ndarray) -> None:
        nonlocal saved, flash_until
        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
        path = args.out / name
        if cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            saved += 1
            flash_until = time.time() + 0.2
            print(f"[{saved}] saved {path}")
        else:
            print(f"ERROR: failed to write {path}")

    cap = open_camera(args.camera, args.exposure)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed, retrying...")
                time.sleep(0.05)
                continue

            now = time.time()
            sharp = sharpness(frame)

            if auto and now >= next_auto:
                if args.min_sharpness and sharp < args.min_sharpness:
                    skipped_blur += 1       # too blurry: try again on the next frame
                else:
                    save(frame)
                    next_auto = now + args.interval

            h, w = frame.shape[:2]
            scale = PREVIEW_WIDTH / w
            preview = cv2.resize(channel_view(frame, view_mode),
                                 (PREVIEW_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

            r, g, b = channel_means(frame)
            put(preview, f"Saved: {saved}   View: {VIEW_NAMES[view_mode]}", 35)
            put(preview, f"Mean R/G/B: {r:.0f} / {g:.0f} / {b:.0f}   Sharpness: {sharp:.0f}", 70,
                (255, 255, 0))
            if auto:
                put(preview, f"AUTO every {args.interval:g}s (skipped blurry: {skipped_blur})", 105,
                    (0, 200, 255))
            if now < flash_until:
                cv2.rectangle(preview, (0, 0), (preview.shape[1] - 1, preview.shape[0] - 1),
                              (0, 255, 0), 8)

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                save(frame)
            elif key == ord("a"):
                auto = not auto
                next_auto = time.time()
                print(f"Auto-capture {'ON' if auto else 'OFF'}")
            elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
                view_mode = int(chr(key))
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
