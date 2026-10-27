#!/usr/bin/env python3
"""Merge your real, annotated photos into dataset_raw.zip (synthetic + real in one zip).

Understands the usual annotation-tool outputs:
    * LabelImg / CVAT / Label Studio "YOLO" export: image + .txt side by side, or
      images/ and labels/ in separate folders (Roboflow-style). Labels are matched by file name.
    * A label file that is empty  = a clean frame (negative sample).
    * An image with NO label file = skipped by default (it may simply not be annotated yet).
      Pass --unlabeled-as-negative ONLY for folders where you have checked every frame is clean.

Validation (the script stops on the first problem so bad labels never reach training):
    * 5 values per line, class id 0 (black_speck), coordinates normalized to 0..1
    * polygon / segmentation labels are rejected (this project uses bounding boxes)

Output is a new zip; the base zip is never modified.

Usage:
    python merge_real_data.py --real foto_asli --base-zip dataset_raw.zip --out dataset_raw_merged.zip
    python merge_real_data.py --real batch1 batch2 --out dataset_raw_merged.zip      # real only
    python merge_real_data.py --real clean_frames --unlabeled-as-negative --base-zip dataset_raw.zip \
        --out dataset_raw_merged.zip
Then upload the result to Drive as MyDrive/qc_plastik/dataset_raw.zip (replace the old one).
"""

import argparse
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
IGNORE_TXT = {"classes.txt", "labels.txt", "notes.txt"}


def check_label(path: Path) -> list[str]:
    lines = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        v = line.split()
        if len(v) != 5:
            raise SystemExit(f"{path}:{n} has {len(v)} values; expected 5 "
                             f"(bounding boxes only, not polygons)")
        if int(float(v[0])) != 0:
            raise SystemExit(f"{path}:{n} class id {v[0]} is invalid; the only class is 0 = black_speck. "
                             f"Check the class order in your annotation tool.")
        if not all(0.0 <= float(x) <= 1.0 for x in v[1:]):
            raise SystemExit(f"{path}:{n} coordinates must be normalized to 0..1")
        lines.append(line.strip())
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, nargs="+", required=True, help="folder(s) with real photos + labels")
    ap.add_argument("--base-zip", type=Path, default=None, help="existing dataset_raw.zip to keep (e.g. synthetic)")
    ap.add_argument("--out", type=Path, default=Path("dataset_raw_merged.zip"))
    ap.add_argument("--unlabeled-as-negative", action="store_true")
    args = ap.parse_args()

    if args.out.resolve() == (args.base_zip.resolve() if args.base_zip else None):
        raise SystemExit("--out must differ from --base-zip")

    # index every label file by stem (so images/ + labels/ layouts work)
    labels: dict[str, Path] = {}
    for folder in args.real:
        for t in folder.rglob("*.txt"):
            if t.name.lower() not in IGNORE_TXT:
                labels.setdefault(t.stem, t)

    used: set[str] = set()
    if args.base_zip:
        with zipfile.ZipFile(args.base_zip) as z:
            used = {Path(n).name for n in z.namelist()}

    entries = []                                 # (image path, label lines)
    n_pos = n_neg = n_skip = 0
    for folder in args.real:
        for img in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS):
            lbl = img.with_suffix(".txt")
            lbl = lbl if lbl.exists() else labels.get(img.stem)
            if lbl is None:
                if not args.unlabeled_as_negative:
                    n_skip += 1
                    continue
                lines = []
            else:
                lines = check_label(lbl)
            entries.append((img, lines))
            n_pos += bool(lines)
            n_neg += not lines

    if not entries:
        raise SystemExit("No usable real images found.")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as out:
        if args.base_zip:
            with zipfile.ZipFile(args.base_zip) as z:
                for info in z.infolist():
                    out.writestr(info.filename, z.read(info.filename))
        for img, lines in entries:
            stem = img.stem
            if stem.startswith("synth_"):         # synthetic prefix is reserved (train-only)
                stem = "real_" + stem
            name, k = stem, 1
            while f"{name}{img.suffix.lower()}" in used:
                name, k = f"{stem}_{k}", k + 1
            used.add(f"{name}{img.suffix.lower()}")
            out.write(img, f"dataset_raw/{name}{img.suffix.lower()}")
            out.writestr(f"dataset_raw/{name}.txt", "\n".join(lines) + ("\n" if lines else ""))

    print(f"Real photos added: {len(entries)}  (with specks: {n_pos}, clean: {n_neg})")
    if n_skip:
        print(f"Skipped {n_skip} images without a label file (not annotated yet?). "
              f"Use --unlabeled-as-negative only if they are verified clean.")
    print(f"Written: {args.out}  ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
