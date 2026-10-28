# QC Plastik Biru - Deteksi Bintik Hitam (black_speck)

Sistem inspeksi visual untuk mendeteksi bintik hitam pada plastik biru gulungan yang bergerak
di mesin cetak. Menggunakan YOLOv8 (Ultralytics) dan OpenCV dengan kamera USB dari atas.
Satu kelas: `black_speck`; gambar bersih dipakai sebagai sampel negatif.

Status: tahap pengembangan (prototipe). Model masih dilatih dari sedikit foto asli
(52 foto) ditambah data sintetis, belum layak sebagai standar produksi.

## Struktur

```
scripts/
  1_capture_dataset.py      ambil foto dari kamera USB atau stream (--camera 0 / URL, --focus, tombol [ ] untuk fokus,
                            s = simpan, a = otomatis, 0-3 = pratinjau kanal, q = keluar)
  annotate.py               alat anotasi kotak YOLO (OpenCV): zoom, c = bersih, x = buang foto, u = batal
  tighten_labels.py         perketat kotak label ke titik gelap yang sebenarnya + lembar pemeriksaan (review/)
  gen_synthetic_dataset.py  dataset sintetis murni (untuk uji alur)
  paste_synthetic.py        sintetis realistis: tempel bintik asli ke foto plastik asli
  merge_real_data.py        gabungkan foto asli beranotasi dengan zip sintetis
  4_realtime_inference.py   deteksi real-time (kamera USB atau URL stream), --focus, tombol s = simpan frame
colab/
  qc_black_speck_colab.ipynb  siapkan dataset, training, evaluasi, simpan best.pt ke Drive
```

Data (zip), bobot (`best.pt`), foto, dan log training disimpan di Google Drive
(`MyDrive/qc_plastik/`), tidak di repo ini (lihat `.gitignore`).

## Alur kerja

1. **Ambil foto asli** dengan kamera dan cahaya final, fokus dikunci:
   `python scripts/1_capture_dataset.py --camera 0 --out foto_asli --focus 120 --interval 3 --min-sharpness 70`
2. **Anotasi:** `python scripts/annotate.py foto_asli` (tarik kotak di bintik; `c` untuk foto bersih).
3. **Perketat label:** `python scripts/tighten_labels.py foto_asli` lalu periksa `foto_asli_tight/review/`
   (kotak berawalan `unclear_` perlu dicek atau dihapus).
4. **Sintetis dari data sendiri:**
   `python scripts/paste_synthetic.py --src foto_asli_tight --n 400 --old-zip dataset_raw_sintetis.zip --old-n 100 --zip`
5. **Gabung:**
   `python scripts/merge_real_data.py --real foto_asli_tight --base-zip synth_paste.zip --out dataset_raw_merged.zip`
6. **Training:** unggah zip gabungan ke `MyDrive/qc_plastik/` sebagai `dataset_raw.zip`, buka notebook di Colab (GPU T4),
   atur `RUN` (nama baru tiap percobaan) dan `IMGSZ = 1280`, Run all. Hasil: `MyDrive/qc_plastik/best.pt`.
7. **Deteksi real-time:**
   `python scripts/4_realtime_inference.py --weights best.pt --imgsz 1280 --conf 0.25 --camera 0 --focus 120`

## Catatan penting

- Kualitas label paling menentukan. Kotak yang jauh lebih besar dari bintiknya menurunkan akurasi tajam
  (mAP50 0,12 pada label longgar menjadi 0,36 setelah diperketat, data dan model sama).
- Foto sintetis (nama berawalan `synth_`) hanya dipakai untuk training. Validasi memakai foto asli.
- `imgsz` saat inferensi harus sama dengan saat training.
- Selalu gabungkan dari zip sintetis murni dengan folder foto asli yang lengkap, agar tidak ada foto yang masuk dua kali.
- Simpan sebagian foto sebagai set uji terpisah (tidak ikut dilatih) untuk menilai model secara jujur.
- Gambar tanpa file label dilewati oleh `merge_real_data.py`, kecuali memakai `--unlabeled-as-negative`
  (hanya untuk folder yang sudah dipastikan bersih). Folder `review/` dan `*_dibuang` diabaikan.

## Rencana pengembangan

- Deteksi pada bintik buram (gerak/fokus): data dengan blur nyata dari kecepatan lini, eksposur pendek atau strobo.
- Deteksi pada plastik yang sudah dicetak: kumpulkan data dari film bercetak, kemungkinan model atau ROI terpisah.
- Penghitungan cacat dan riwayat: pelacakan objek antar-frame agar satu bintik tidak terhitung berkali-kali,
  catatan per roll/meter (CSV atau SQLite) berisi waktu, posisi, skor, dan potongan gambar.
- Uji ukuran: tetapkan ukuran bintik terkecil, tingkat deteksi, dan batas deteksi palsu bersama pihak QC.

## Instalasi

```
pip install -r requirements.txt
```