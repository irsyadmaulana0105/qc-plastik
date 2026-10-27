#!/usr/bin/env python3
"""Real-time black_speck detection on a USB camera using a trained YOLOv8 model.

Usage:
    python 4_realtime_inference.py
    python 4_realtime_inference.py --weights best.pt --conf 0.35 --imgsz 640 --camera 0

Keys:
    q - quit
"""

import argparse
import sys
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
PREVIEW_WIDTH = 1280
WINDOW_NAME = "QC - black_speck detection  [q] quit"
BOX_COLOR = (0, 0, 255)  # BGR


class LatestFrameReader:
    """Grabs frames in a background thread so inference always sees the newest frame
    (avoids the stale-buffer lag that appears when inference is slower than the camera)."""

    def __init__(self, index: int):
        if sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        elif sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY

        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened at {w}x{h}")

        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            frame = self._frame
            self._frame = None  # each frame is consumed once
        return frame

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def draw_detections(frame, result, names) -> int:
    """Draw boxes, class labels and confidence on `frame` (in place). Returns detection count."""
    h, w = frame.shape[:2]
    thickness = max(2, round(min(h, w) / 400))
    font_scale = max(0.5, min(h, w) / 1200)

    boxes = result.boxes
    for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(),
                               boxes.conf.cpu().numpy(),
                               boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[int(cls)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thickness)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = y1 - 6 if y1 - th - baseline - 6 > 0 else y2 + th + 6
        cv2.rectangle(frame, (x1, ty - th - baseline), (x1 + tw, ty + baseline), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
    return len(boxes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"
    print(f"Device: {'cuda:0' if use_cuda else 'cpu'}")

    model = YOLO(args.weights)
    names = model.names

    # Warm-up so the first displayed FPS value is not skewed by CUDA initialization
    dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=args.imgsz, device=device, verbose=False)

    reader = LatestFrameReader(args.camera)
    fps, prev_t = 0.0, time.perf_counter()

    try:
        while True:
            frame = reader.read()
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.001)
                continue

            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                verbose=False,
            )[0]

            n = draw_detections(frame, result, names)

            now = time.perf_counter()
            inst = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst  # smoothed

            scale = PREVIEW_WIDTH / frame.shape[1]
            view = cv2.resize(frame, (PREVIEW_WIDTH, int(frame.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
            cv2.putText(view, f"FPS: {fps:.1f}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(view, f"Detections: {n}", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if n == 0 else (0, 0, 255),
                        2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
