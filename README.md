# QC Plastik Biru - Deteksi Bintik Hitam (black_speck)

Sistem inspeksi visual untuk mendeteksi bintik hitam pada plastik biru gulungan yang bergerak
di mesin cetak. Menggunakan YOLOv8 (Ultralytics) dan OpenCV dengan kamera USB dari atas dan
backlight di bawah plastik. Satu kelas: `black_speck`; gambar bersih dipakai sebagai sampel negatif.

## Struktur

```
scripts/
  1_capture_dataset.py      ambil foto dari kamera (s = simpan, a = otomatis, 0-3 = pratinjau kanal, q = keluar)
  gen_synthetic_dataset.py  buat dataset sintetis + label YOLO otomatis (untuk uji alur / pra-latih)
  merge_real_data.py        gabungkan foto asli beranotasi ke dataset_raw.zip
  4_realtime_inference.py   deteksi real-time dari kamera memakai best.pt
colab/
  qc_black_speck_colab.ipynb  siapkan dataset, training, evaluasi, simpan best.pt ke Drive
```

Data (`dataset_raw.zip`), bobot (`best.pt`), dan log training disimpan di Google Drive
(`MyDrive/qc_plastik/`), tidak di repo ini (lihat `.gitignore`).

## Alur kerja

1. **Data uji (sintetis):** `python scripts/gen_synthetic_dataset.py --n 500 --seed 7 --zip dataset_synth.zip`
2. **Data asli:** ambil foto di mesin dengan `scripts/1_capture_dataset.py --interval 2`, anotasi bounding box
   (LabelImg / CVAT, format YOLO, kelas 0), lalu gabungkan:
   `python scripts/merge_real_data.py --real foto_asli --base-zip dataset_synth.zip --out dataset_raw.zip`
3. **Training:** upload `dataset_raw.zip` ke `MyDrive/qc_plastik/`, buka notebook di Google Colab (GPU T4), Run all.
   Hasil: `MyDrive/qc_plastik/best.pt`.
4. **Deteksi real-time:** `python scripts/4_realtime_inference.py --weights best.pt --imgsz 640 --conf 0.35`

Catatan penting:
- Foto sintetis (nama berawalan `synth_`) hanya dipakai untuk training. Validasi memakai foto asli.
- `imgsz` saat inferensi harus sama dengan saat training. Jika speck sangat kecil, gunakan `IMGSZ = 1280` di notebook.
- Selalu gabungkan dari zip sintetis murni dengan folder foto asli yang lengkap, agar tidak ada foto yang masuk dua kali.
- Gambar tanpa file label dilewati oleh `merge_real_data.py`, kecuali memakai `--unlabeled-as-negative`
  (hanya untuk folder yang sudah dipastikan bersih).

## Instalasi

```
pip install -r requirements.txt
```
