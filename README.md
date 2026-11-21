# Kasus 08 · Prediksi Keterlambatan Paket

Model klasifikasi biner untuk memperkirakan apakah paket akan terlambat, memakai
data saat pickup dari hub Bandung. Metrik utama adalah **PR-AUC**.

## Anggota

| Nama | NIM | Bagian |
|---|---:|---|
| Roven Angger Djuha Al Haq | 2401168 | EDA, pembersihan data, baseline |
| Rafli Maulana | 2404364 | Fitur, perbandingan model, kalibrasi/threshold |
| Sandy Arya Wiguna | 2401084 | API, aplikasi Streamlit, deploy |

## Struktur repositori

```text
notebooks/   01_cleaning, 02_eda, 03_baseline, 04_models, 05_calibration_threshold, 06_final_test, 07_post_test_analysis
data/        raw.csv (mentah, read-only), clean.csv (hasil bersih)
src/         features.py (clean_raw, build_features, siapkan_fitur)
artifacts/   model.joblib, metadata.json, thresholds.json, model_base.joblib
app/api/     main.py (FastAPI)
app/ui/      streamlit_app.py
reports/     figures/
```

## Menjalankan

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py
```

## Data dan pembersihan

`data/raw.csv`: 40.500 baris, 13 kolom. SHA-256:
`2821b50eac1b02181cacbfb52fca784b2b5be930c73e5c671a877eeec6a600ab`.

Split berdasarkan waktu: train sampai 31 Desember 2025 (26.633 baris),
validation Januari-Maret 2026 (6.623 baris), test April-Juni 2026 (6.744 baris).
Test dibuka satu kali pada tahap akhir.

| Masalah | Baris | Penanganan | Alasan |
|---|---:|---|---|
| Duplikat identik | 500 | Hapus dengan `drop_duplicates` | Isi sama persis; bobot ganda jika dibiarkan |
| `id_resi` ganda | 1.000 baris / 500 grup | Terhapus bersama duplikat | 0 konflik nilai; tersisa 40.000 id unik |
| Label `layanan` tidak baku | 589 (812 mentah) | Petakan ke EXPRESS/REG/HEMAT | Hanya beda kapitalisasi/singkatan |
| Ejaan `kota_tujuan` tidak baku | 1.132 (1.148 mentah) | Petakan dengan daftar alias | 28 ejaan menjadi 15 kota; Bali dipetakan ke Denpasar karena jaraknya identik |
| `jarak_km` kosong | 800 (813 mentah) | Biarkan NaN + flag | Imputasi hanya dari train |
| `cuaca_saat_pickup` kosong | 400 (406 mentah) | Biarkan NaN + flag | "Tidak diketahui" bermakna; imputasi tunggu train |
| `berat_kg` tidak wajar | 1.274 (1.286 mentah) | NaN + `flag_berat_invalid` | 81 negatif, 1.205 > 100 kg; kemungkinan gram/salah input |
| `janji_hari` tidak sesuai layanan | 282 | Flag saja | EXPRESS 1-3, REG 2-4, HEMAT 4-6; bisa SLA khusus |
| Jam, tanggal, jarak nol/negatif, spasi | 0 | Tidak diubah | Sudah valid; API tetap memvalidasi |
| Nilai hilang terkait target | jarak 15,9% vs 17,8%; cuaca 19,2% vs 17,8% | Tidak ada penanganan khusus | Selisih tingkat terlambat kecil |
| `volume_hub_hari` sebagai fitur | tidak dipakai | Tidak dipakai di `build_features` | Total harian bisa baru lengkap setelah pickup; berisiko bocor |

## Hasil

Validasi (Januari-Maret 2026), model final sebelum kalibrasi/threshold:

| Model | PR-AUC val | ROC-AUC val |
|---|---:|---:|
| Logistic Regression original features (termasuk `volume_hub_hari`, leaky, tidak deploy) | 0,6458 | 0,8311 |
| Logistic Regression + lag features | 0,5913 | 0,8111 |
| Random Forest | 0,5624 | 0,7951 |
| LightGBM | 0,5785 | 0,7982 |

Ablasi `leaky, not deployed - Logistic Regression dengan volume_hub_hari` = PR-AUC 0,6458.

Kalibrasi: sigmoid OOF Brier 0,1282 vs raw 0,1925 vs isotonic 0,1292. Dipakai sigmoid.

Test (April-Juni 2026), dibuka satu kali:

| Model | PR-AUC test | ROC-AUC test |
|---|---:|---:|
| A. LogReg original incl. volume (leaky, not deployable) | 0,3560 | 0,7452 |
| B. LogReg original tanpa volume | 0,3601 | 0,7471 |
| C. Final LogReg+lag sigmoid | **0,3608** | **0,7502** |

Model final pada test: notify threshold 0,090 → precision 0,2039, recall 0,8328,
share notified 56,87%, biaya Rp23.115.000 vs nobody Rp46.950.000 dan everyone
Rp29.025.000. Prioritas threshold 0,523 → share 3,05%, precision 0,5485,
recall 0,1203, lift 3,94. Brier test 0,1053. PR-AUC per bulan test:
Apr 0,3593, Mei 0,3475, Jun 0,3904.

## Threshold dan alasan bisnis

- Biaya terlambat tanpa kabar: Rp50.000.
- Biaya kabar keliru: Rp5.000.
- Titik impas teoretis: `5.000 / (50.000 + 5.000) ≈ 0,0909`.
- Threshold final: `kabari pelanggan = 0,090`, `prioritaskan = 0,523`.
- Batas kabari dipilih dari biaya minimum pada validation: Rp21.170.000 vs
  Rp25.870.000 (kabari semua) dan Rp72.450.000 (tidak mengabari siapa pun).
- Batas prioritaskan dipilih dari porsi top 10% probabilitas, dengan precision
  0,6863, recall 0,3140, lift 3,137. Zona ini lebih kecil dari zona kabari.
- Threshold dikunci sebelum test dibuka.

## Aplikasi

- API: `app/api/main.py` (FastAPI), endpoint `POST /predict`, validasi Pydantic,
  `GET /health`, model dimuat sekali saat startup.
- Antarmuka: `app/ui/streamlit_app.py`, form input, prediksi, rekomendasi berwarna,
  batch CSV, fallback model lokal jika API gagal.
- Deploy: API di layanan gratis (Render/Railway/Hugging Face Spaces), Streamlit di
  Streamlit Community Cloud. Alamat API diatur lewat `API_URL` di secrets/env.

## Keterbatasan

- Model final hampir tidak menang dari baseline tanpa `volume_hub_hari`; sinyal
  kuat berasal dari volume harian yang berisiko bocor.
- Threshold dan kalibrasi dihitung pada validation Januari-Maret 2026; drift
  (Maret tinggi, November-Desember tinggi) bisa memengaruhi cost.
- Fitur lag membutuhkan volume hub hari sebelumnya; tanggal awal tidak punya lag.
- `volume_hub_hari` tidak konstan dalam satu tanggal; median harian dipakai.
- Data sintetis; interpretasi operasional perlu divalidasi di data nyata.

## Tautan

- Repositori: `https://github.com/venever1/parcel-delay-prediction`
- Streamlit: `https://parcel-delay-prediction.streamlit.app/`
- Video: `<tautan video>`
Note : API tidak dikonfigurasi dijalankan di local

## Bagian yang dibantu AI

Agen AI membantu penulisan kode, notebook, dan README; seluruh keputusan
cleaning, fitur, model, dan threshold diverifikasi lewat eksekusi nyata.
