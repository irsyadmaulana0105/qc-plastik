#!/usr/bin/env python3
"""Mini YOLO box annotator for black_speck (OpenCV only, no extra install).

Writes one YOLO .txt next to every image (class 0 = black_speck), the same format
merge_real_data.py expects. Built for tiny specks: big zoom, thin boxes, nearest-neighbour pixels.

Usage:
    python scripts/annotate.py foto_asli
    python scripts/annotate.py foto_uji --view-w 1400 --view-h 800

Mouse:
    left-drag   draw a box around a speck
    wheel       zoom in/out around the cursor     (or keys  =  and  -  )
    right-drag  pan the image while zoomed

Keys:
    d  next image (saves)        a  previous image (saves)
    n  jump to next UNMARKED image
    u  undo last box             r  remove ALL boxes of this image
    c  mark image as CLEAN (no specks; writes an empty .txt = negative sample)
    x  DISCARD this image (blurry/bad): moved with its label to <folder>_dibuang/ next to the folder,
       not deleted, so you can move it back
    f  fit whole image to window
    q  save and quit

Status shown on screen:  BOXES n  /  CLEAN  /  UNMARKED (no .txt yet, skipped by merge_real_data.py)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
WIN = "Anotasi black_speck  [d/a] next/prev  [n] unmarked  [u] undo  [c] clean  [x] buang  [f] fit  [q] quit"
MIN_BOX_PX = 2          # ignore accidental clicks smaller than this (image pixels)
BOX_COLOR = (0, 0, 255)
DRAW_COLOR = (0, 255, 255)


def read_yolo(txt: Path, w: int, h: int):
    """Return (boxes in pixel xyxy, exists)."""
    if not txt.exists():
        return [], False
    boxes = []
    for line in txt.read_text().splitlines():
        v = line.split()
        if len(v) != 5:
            continue
        cx, cy, bw, bh = (float(x) for x in v[1:])
        boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes, True


def write_yolo(txt: Path, boxes, w: int, h: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    txt.write_text("\n".join(lines) + ("\n" if lines else ""))


class Annotator:
    def __init__(self, folder: Path, view_w: int, view_h: int):
        self.paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.paths:
            raise SystemExit(f"No images found in {folder}")
        self.vw, self.vh = view_w, view_h
        self.i = 0
        self.img = None
        self.boxes = []
        self.clean = False
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self.drawing = False
        self.start = self.cur = (0.0, 0.0)
        self.panning = False
        self.pan_last = (0, 0)
        self.mouse = (0, 0)

    # ---- file handling -------------------------------------------------------------
    def txt(self, i=None) -> Path:
        return self.paths[self.i if i is None else i].with_suffix(".txt")

    def marked(self, i: int) -> bool:
        return self.txt(i).exists()

    def load(self, i: int) -> None:
        self.i = i
        self.img = cv2.imread(str(self.paths[i]))
        if self.img is None:
            raise SystemExit(f"Cannot read {self.paths[i]}")
        h, w = self.img.shape[:2]
        self.boxes, exists = read_yolo(self.txt(), w, h)
        self.clean = exists and not self.boxes
        self.fit()

    def save(self) -> None:
        h, w = self.img.shape[:2]
        if self.boxes or self.clean:
            write_yolo(self.txt(), self.boxes, w, h)
        elif self.txt().exists():          # all boxes removed and not marked clean -> unmarked again
            self.txt().unlink()

    def go(self, i: int) -> None:
        self.save()
        self.load(max(0, min(len(self.paths) - 1, i)))

    def discard(self) -> None:
        """Move the current image (and its label) out of the folder; never deletes anything."""
        p = self.paths[self.i]
        dest = p.parent.parent / (p.parent.name + "_dibuang")
        dest.mkdir(exist_ok=True)
        p.replace(dest / p.name)
        t = p.with_suffix(".txt")
        if t.exists():
            t.replace(dest / t.name)
        print(f"Dibuang: {p.name} -> {dest}")
        del self.paths[self.i]
        if not self.paths:
            raise SystemExit("Semua foto sudah dibuang.")
        self.load(min(self.i, len(self.paths) - 1))

    def next_unmarked(self) -> None:
        self.save()
        n = len(self.paths)
        for k in range(1, n + 1):
            j = (self.i + k) % n
            if not self.marked(j):
                self.load(j)
                return
        print("Semua gambar sudah ditandai.")
        self.load(self.i)

    # ---- view math -----------------------------------------------------------------
    def fit(self) -> None:
        h, w = self.img.shape[:2]
        self.scale = min(self.vw / w, self.vh / h)
        self.ox = self.oy = 0.0

    def to_img(self, vx, vy):
        return vx / self.scale + self.ox, vy / self.scale + self.oy

    def to_view(self, ix, iy):
        return (ix - self.ox) * self.scale, (iy - self.oy) * self.scale

    def zoom(self, factor: float, vx: float, vy: float) -> None:
        ix, iy = self.to_img(vx, vy)
        self.scale = float(np.clip(self.scale * factor, 0.05, 60.0))
        self.ox, self.oy = ix - vx / self.scale, iy - vy / self.scale

    # ---- mouse ---------------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, _param=None) -> None:
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing, self.start, self.cur = True, self.to_img(x, y), self.to_img(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.cur = self.to_img(x, y)
            if self.panning:
                dx, dy = x - self.pan_last[0], y - self.pan_last[1]
                self.ox -= dx / self.scale
                self.oy -= dy / self.scale
                self.pan_last = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.start
            x2, y2 = self.to_img(x, y)
            if abs(x2 - x1) >= MIN_BOX_PX and abs(y2 - y1) >= MIN_BOX_PX:
                self.boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                self.clean = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.panning, self.pan_last = True, (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.panning = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except AttributeError:
                delta = 1 if flags > 0 else -1
            self.zoom(1.25 if delta > 0 else 0.8, x, y)

    # ---- drawing -------------------------------------------------------------------
    def render(self) -> np.ndarray:
        m = np.float32([[self.scale, 0, -self.ox * self.scale],
                        [0, self.scale, -self.oy * self.scale]])
        interp = cv2.INTER_NEAREST if self.scale >= 2 else cv2.INTER_LINEAR
        view = cv2.warpAffine(self.img, m, (self.vw, self.vh), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
        for x1, y1, x2, y2 in self.boxes:
            a, b = self.to_view(x1, y1), self.to_view(x2, y2)
            cv2.rectangle(view, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), BOX_COLOR, 1)
        if self.drawing:
            a, b = self.to_view(*self.start), self.to_view(*self.cur)
            cv2.rectangle(view, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), DRAW_COLOR, 1)

        status = (f"BOXES {len(self.boxes)}" if self.boxes
                  else "CLEAN" if self.clean else "UNMARKED")
        done = sum(self.marked(j) for j in range(len(self.paths)))
        lines = [f"{self.paths[self.i].name}   [{self.i + 1}/{len(self.paths)}]   {status}",
                 f"zoom x{self.scale:.1f}   marked {done}/{len(self.paths)}"]
        for k, t in enumerate(lines):
            y = 26 + 26 * k
            cv2.putText(view, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if self.boxes else (0, 200, 255), 2, cv2.LINE_AA)
        return view

    # ---- main loop -----------------------------------------------------------------
    def run(self) -> None:
        first = next((j for j in range(len(self.paths)) if not self.marked(j)), 0)
        self.load(first)
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKey(15) & 0xFF
            if k == 255:
                continue
            c = chr(k).lower() if 32 <= k < 127 else ""
            if c == "d":
                self.go(self.i + 1)
            elif c == "a":
                self.go(self.i - 1)
            elif c == "n":
                self.next_unmarked()
            elif c == "u" and self.boxes:
                self.boxes.pop()
            elif c == "r":
                self.boxes.clear()
            elif c == "c":
                self.boxes.clear()
                self.clean = True
            elif c == "x":
                self.discard()
            elif c == "f":
                self.fit()
            elif c in ("=", "+"):
                self.zoom(1.25, self.vw / 2, self.vh / 2)
            elif c == "-":
                self.zoom(0.8, self.vw / 2, self.vh / 2)
            elif c == "q":
                break
        self.save()
        cv2.destroyAllWindows()
        n_box = sum(1 for j in range(len(self.paths)) if self.marked(j) and self.txt(j).stat().st_size > 0)
        n_clean = sum(1 for j in range(len(self.paths)) if self.marked(j) and self.txt(j).stat().st_size == 0)
        left = sum(1 for j in range(len(self.paths)) if not self.marked(j))
        print(f"Selesai. Dengan bintik: {n_box} | bersih: {n_clean} | belum ditandai: {left}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini YOLO annotator for black_speck")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--view-w", type=int, default=1500, help="window width in pixels")
    ap.add_argument("--view-h", type=int, default=850, help="window height in pixels")
    args = ap.parse_args()
    Annotator(args.folder, args.view_w, args.view_h).run()


if __name__ == "__main__":
    main()