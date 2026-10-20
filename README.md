# ForestWatch AI — Perbandingan Sentinel-1 & Sentinel-2 untuk Perubahan Tutupan Hutan (Sesi 6)

## 1. Identitas Proyek

| Item | Keterangan |
|---|---|
| Nama Proyek | ForestWatch AI |
| Sesi | 6 dari 6 |
| Judul Penelitian | Analisis Perbandingan Citra Sentinel-1 dan Sentinel-2 untuk Klasifikasi Perubahan Tutupan Hutan di Provinsi Bengkulu |
| Jenis | Klasifikasi tutupan hutan + analisis perubahan (radar vs optik) |

## 2. Latar Belakang

Pemantauan perubahan tutupan hutan penting untuk lingkungan dan tata kelola lahan.
Sentinel-1 (radar) mampu menembus awan, sedangkan Sentinel-2 (optik) kaya informasi
spektral. Proyek ini membandingkan performa klasifikasi tutupan hutan dari kedua sensor.

## 3. Tujuan

1. Memuat citra Sentinel-1 dan Sentinel-2 serta memeriksa metadata/kanal/resolusi.
2. Menangani NoData dan nilai raster tidak valid.
3. Mengekstraksi fitur sesuai sensor (radar: VV/VH; optik: kanal spektral + NDVI).
4. Mengklasifikasi tutupan hutan dan membandingkan hasil kedua sensor.
5. Menghitung metrik evaluasi bila label referensi tersedia.
6. Menganalisis perubahan tutupan hutan dan mengekspor hasil.

## 4. Teknologi

Python, Rasterio, GeoPandas, NumPy, Pandas, scikit-learn, Streamlit, Plotly, Joblib, pytest.

## 5. Sumber Dataset

- **Data nyata:** GeoTIFF Sentinel-1 (VV/VH) dan Sentinel-2 (B02–B08) dengan label
  referensi. Data dapat diunduh dari Copernicus Open Access Hub / Google Earth Engine.
- **Mode demo:** dataset sintetik biner (bukan hutan/hutan) untuk S1 dan S2.

> **Catatan penting:**
> - Data demo sintetik **bukan** citra Sentinel nyata; hasilnya hanya validasi pipeline.
> - **Tidak semua perubahan tutupan hutan disebabkan aktivitas manusia.** Klasifikasi
>   perubahan tutupan hutan **tidak otomatis** sama dengan klasifikasi penyebab deforestasi.
> - Sentinel-1 (radar) dan Sentinel-2 (optik) diperlakukan sebagai sensor berbeda dengan
>   fitur dan interpretasi masing-masing.

## 6. Struktur Folder

```
Sesi-6-Forest-Change-Analysis/
├── app.py
├── train.py
├── requirements.txt
├── config/kelas.py            # Definisi kelas & warna perubahan
├── src/
│   ├── data_loader.py         # Baca raster, fitur S1/S2, NDVI, demo sintetik
│   └── model_ml.py            # Random Forest, perbandingan S1 vs S2, peta perubahan
├── tests/
├── data/
├── models/                    # model_s1.joblib, model_s2.joblib
└── outputs/results/
```

## 7. Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Cara Menjalankan

```bash
python train.py          # latih & bandingkan S1 vs S2 (demo)
streamlit run app.py     # antarmuka
```

## 9. Cara Melatih Model

`python train.py` melatih dua Random Forest (satu untuk S1, satu untuk S2) pada data demo,
mengevaluasi, dan menyimpan model ke `models/model_s1.joblib` & `model_s2.joblib` beserta
laporan perbandingan di `outputs/results/laporan_perbandingan.json`. Untuk data nyata,
berikan fitur dari `src.data_loader.fitur_sentinel1/2` ke `bandingkan_s1_s2()`.

## 10. Cara Evaluasi

Metrik (accuracy, precision, recall, F1, confusion matrix) dihitung pada data uji
terpisah. Perbandingan S1 vs S2 ditampilkan di aplikasi.

## 11. Contoh Hasil (mode demo)

| Sensor | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| S1 (radar) | 0.963 | 0.964 | 0.963 | 0.963 |
| S2 (optik) | 0.995 | 0.995 | 0.995 | 0.995 |

> Nilai dari data sintetik; hanya menunjukkan pipeline berfungsi dan S2 lebih unggul pada
> data demo. Pada data nyata hasil sangat bergantung wilayah, musim, dan label referensi.

## 12. Keterbatasan

- Hasil demo dari data sintetik, bukan citra nyata.
- Analisis perubahan tidak menyimpulkan penyebab (deforestasi) — hanya perubahan tutupan.
- Model CNN (bila dipakai) harus menyesuaikan jumlah kanal input sensor.
- Belum ada penyesuaian koordinat/resolusi lintas sensor otomatis pada mode unggah.

## 13. Pengujian

```bash
python -m pytest tests -q
```

## 14. Referensi

- Copernicus Sentinel-1 & Sentinel-2 (ESA).
- Breiman, L. (2001). Random Forests.
- Analisis perubahan tutupan lahan berbasis citra multisensor.
