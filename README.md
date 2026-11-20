# Kasus 08 · Prediksi Keterlambatan Paket

Model klasifikasi biner untuk memperkirakan apakah paket akan terlambat, memakai data saat pickup.

Status: placeholder. README lengkap menyusul setelah Tahap A.

## Struktur

```
notebooks/   eksplorasi, pembersihan, pelatihan model
data/        raw.csv dan data bersih
src/         fitur yang dipakai saat latihan dan prediksi
artifacts/   model.joblib dan metadata.json
app/api/     API FastAPI
app/ui/      antarmuka Streamlit
```

## Menjalankan

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py
```
