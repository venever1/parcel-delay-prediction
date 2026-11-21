"""Antarmuka Streamlit untuk prediksi keterlambatan paket.

API dipakai sebagai jalur utama. Jika API tidak dapat dihubungi, aplikasi
memakai model lokal sebagai cadangan agar tetap bisa dipakai.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.main import PackageInput
import sys
from pathlib import Path

# taruh root repo di urutan pertama agar `src` milik proyek ini yang terbaca
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features import KOTA_ALIAS, siapkan_fitur

st.set_page_config(page_title="Prediksi Keterlambatan Paket", layout="wide")

KOLOM_INPUT = [
    "tanggal_pickup",
    "jam_pickup",
    "kota_tujuan",
    "jarak_km",
    "jumlah_transit",
    "layanan",
    "janji_hari",
    "berat_kg",
    "metode_bayar",
    "cuaca_saat_pickup",
    "volume_hub_hari",
    "volume_hub_kemarin",
    "volume_hub_7hari",
]


def api_url() -> str | None:
    """Ambil alamat API dari secrets atau environment variable."""
    try:
        nilai = st.secrets.get("API_URL")
    except Exception:
        nilai = None
    if not nilai:
        nilai = os.getenv("API_URL")
    if not nilai:
        return None
    return str(nilai).rstrip("/")


@st.cache_resource
def muat_model_cadangan():
    """Muat model_once saat fallback diperlukan."""
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    metadata = json.loads((ROOT / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def rekomendasi(probabilitas: float, thresholds: dict) -> str:
    if probabilitas >= float(thresholds["prioritaskan"]):
        return "prioritaskan"
    if probabilitas >= float(thresholds["kabari_pelanggan"]):
        return "kabari pelanggan"
    return "aman"


def pesan_galat(respons: requests.Response) -> str:
    """Ubah error API menjadi kalimat yang mudah dibaca."""
    try:
        isi = respons.json()
    except ValueError:
        return f"API mengembalikan galat {respons.status_code}."

    detail = str(isi.get("detail", "Data ditolak API."))
    errors = isi.get("errors", [])
    if errors:
        ringkas = []
        for err in errors[:3]:
            ringkas.append(f"{err.get('field', 'input')}: {err.get('message', 'tidak valid')}")
        detail += ". " + "; ".join(ringkas)
    return detail


def prediksi_lokal(payload: dict) -> dict:
    """Fallback: prediksi memakai model lokal dengan validasi yang sama."""
    try:
        PackageInput.model_validate(payload)
    except ValidationError as exc:
        ringkas = []
        for err in exc.errors()[:3]:
            lokasi = ".".join(str(bagian) for bagian in err["loc"])
            ringkas.append(f"{lokasi}: {err['msg']}")
        return {"error": "Data tidak valid. " + "; ".join(ringkas)}

    try:
        model, metadata = muat_model_cadangan()
        data = pd.DataFrame([payload])
        fitur = siapkan_fitur(data, metadata)
        probabilitas = float(model.predict_proba(fitur)[0, 1])
        return {
            "probabilitas": probabilitas,
            "rekomendasi": rekomendasi(probabilitas, metadata["thresholds"]),
            "model_version": metadata["model_version"],
        }
    except Exception:
        return {"error": "Model cadangan belum tersedia."}


def prediksi(payload: dict, tampilkan_cadangan: bool = True) -> dict:
    """Pakai API jika dikonfigurasi; jika gagal, jatuh ke model lokal."""
    url = api_url()
    if not url:
        return prediksi_lokal(payload)

    try:
        respons = requests.post(f"{url}/predict", json=payload, timeout=10)
        if respons.status_code == 200:
            return respons.json()
        return {"error": pesan_galat(respons)}
    except (requests.RequestException, ValueError):
        if tampilkan_cadangan:
            st.caption("Prediksi memakai model lokal")
        return prediksi_lokal(payload)


def tampilkan_hasil(hasil: dict) -> None:
    if "error" in hasil:
        st.error(f"Maaf, prediksi gagal: {hasil['error']}")
        return

    probabilitas = float(hasil["probabilitas"])
    rekom = hasil["rekomendasi"]
    st.session_state["probabilitas_mentah"] = probabilitas
    st.metric("Probabilitas terlambat", f"{probabilitas:.1%}")
    st.caption(f"Versi model: {hasil['model_version']}")

    if rekom == "prioritaskan":
        st.error("Rekomendasi: PRIORITASKAN paket ini.")
    elif rekom == "kabari pelanggan":
        st.warning("Rekomendasi: KABARI PELANGGAN lebih awal.")
    else:
        st.success("Rekomendasi: AMAN, tidak perlu dikabari.")


def nilai_csv(nilai) -> object:
    """Ubah sel CSV menjadi nilai JSON; sel kosong dikirim sebagai null."""
    if pd.isna(nilai) or str(nilai).strip() == "":
        return None
    if hasattr(nilai, "date") and callable(nilai.date):
        return nilai.date().isoformat()
    if hasattr(nilai, "isoformat") and callable(nilai.isoformat):
        return nilai.isoformat()
    return str(nilai).strip()


st.title("Prediksi Keterlambatan Paket")
_status_api = api_url() or "tidak dikonfigurasi (mode lokal)"
st.caption(f"API: {_status_api} | Model: {muat_model_cadangan()[1]['model_version']}")

tab_satu, tab_batch = st.tabs(["Prediksi Satu Paket", "Prediksi Batch CSV"])

with tab_satu:
    with st.form("form_paket"):
        kiri, kanan = st.columns(2)
        with kiri:
            tanggal = st.date_input("Tanggal pickup", value=date(2025, 6, 15))
            jam = st.slider("Jam pickup", min_value=7, max_value=22, value=9)
            kota = st.selectbox(
                "Kota tujuan",
                sorted(set(KOTA_ALIAS.values())),
            )
            jarak = st.number_input("Jarak dari Bandung (km)", min_value=0.0, max_value=3000.0, value=150.0)
            transit = st.number_input("Jumlah transit", min_value=0, max_value=10, value=1)
        with kanan:
            layanan = st.selectbox("Layanan", ["REG", "EXPRESS", "HEMAT"])
            janji = st.number_input("Janji tiba (hari)", min_value=1, max_value=14, value=4)
            berat = st.number_input("Berat (kg)", min_value=0.0, max_value=100.0, value=1.2, step=0.1)
            metode = st.selectbox("Metode bayar", ["COD", "Non COD"])
            cuaca = st.selectbox("Cuaca saat pickup", ["Cerah", "Berawan", "Hujan", "Hujan Lebat"])
            volume = st.number_input("Volume hub hari itu", min_value=0, max_value=30000, value=4800)
            volume_kemarin = st.number_input("Volume hub kemarin", min_value=0, max_value=30000, value=5000)
            volume_7hari = st.number_input("Rata-rata volume hub 7 hari sebelumnya", min_value=0.0, max_value=30000.0, value=5100.0)

        dikirim = st.form_submit_button("Prediksi")

    if dikirim:
        if tanggal is None:
            st.error("Tanggal pickup belum diisi.")
        else:
            payload = {
                "tanggal_pickup": tanggal.isoformat(),
                "jam_pickup": int(jam),
                "kota_tujuan": kota,
                "jarak_km": float(jarak),
                "jumlah_transit": int(transit),
                "layanan": layanan,
                "janji_hari": int(janji),
                "berat_kg": float(berat),
                "metode_bayar": metode,
                "cuaca_saat_pickup": cuaca,
                "volume_hub_hari": int(volume),
                "volume_hub_kemarin": int(volume_kemarin),
                "volume_hub_7hari": float(volume_7hari),
            }
            with st.spinner("Menghitung probabilitas..."):
                tampilkan_hasil(prediksi(payload))

with tab_batch:
    st.write("Unggah CSV dengan kolom sesuai form di atas. Baris kosong atau tidak valid akan ditandai.")
    berkas = st.file_uploader("Pilih CSV", type=["csv"])
    if berkas is not None:
        try:
            tabel = pd.read_csv(berkas)
        except Exception:
            st.error("CSV tidak dapat dibaca. Periksa kembali format file.")
            st.stop()

        hilang = [kolom for kolom in KOLOM_INPUT if kolom not in tabel.columns]
        if hilang:
            st.error(f"Kolom belum ada: {', '.join(hilang)}")
        else:
            hasil_baris = []
            for indeks, baris in tabel.iterrows():
                payload = {kolom: nilai_csv(baris[kolom]) for kolom in KOLOM_INPUT}
                hasil = prediksi(payload, tampilkan_cadangan=(indeks == 0))
                hasil_baris.append(
                    {
                        "baris": int(indeks) + 1,
                        "probabilitas": hasil.get("probabilitas"),
                        "rekomendasi": hasil.get("rekomendasi", ""),
                        "error": hasil.get("error", ""),
                    }
                )

            keluaran = pd.concat([tabel.reset_index(drop=True), pd.DataFrame(hasil_baris)], axis=1)
            st.dataframe(keluaran, use_container_width=True)
            st.download_button(
                "Unduh hasil prediksi",
                data=keluaran.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi.csv",
                mime="text/csv",
            )
