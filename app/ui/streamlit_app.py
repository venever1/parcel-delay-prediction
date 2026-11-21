"""Antarmuka Streamlit untuk prediksi keterlambatan paket.

API dipakai sebagai jalur utama. Jika API tidak dapat dihubungi, aplikasi
memakai model lokal sebagai cadangan agar tetap bisa dipakai.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
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
from src.features import KOTA_ALIAS, siapkan_fitur

st.set_page_config(page_title="Prediksi Keterlambatan Paket", layout="wide")

KOLOM_INPUT = [
    "Tanggal Pickup",
    "Jam Pickup",
    "Kota Tujuan",
    "Jarak Km",
    "Jumlah Transit",
    "layanan",
    "Janji Hari",
    "Berat Kg",
    "Metode Bayar",
    "Cuaca Saat Pickup",
    "Volume hub Kemarin",
    "Volume Hub 7Hari",
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
    """Muat model sekali saat fallback diperlukan."""
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


def tema_aktif() -> str:
    """Deteksi tema aktif; fallback ke light jika tidak tersedia."""
    try:
        nilai = st.context.theme.type
    except Exception:
        nilai = None
    return "dark" if nilai == "dark" else "light"


def gaya_zona(rekom: str, tema: str) -> dict:
    """Warna dan ikon untuk zona rekomendasi."""
    if tema == "dark":
        return {
            "aman": {"aksen": "#34D399", "latar": "#0F2F25", "teks": "#A7F3D0", "ikon": "✓"},
            "kabari pelanggan": {"aksen": "#FBBF24", "latar": "#3A2A0A", "teks": "#FDE68A", "ikon": "!"},
            "prioritaskan": {"aksen": "#F87171", "latar": "#3B1414", "teks": "#FECACA", "ikon": "!"},
        }[rekom]
    return {
        "aman": {"aksen": "#059669", "latar": "#D1FAE5", "teks": "#065F46", "ikon": "✓"},
        "kabari pelanggan": {"aksen": "#D97706", "latar": "#FEF3C7", "teks": "#92400E", "ikon": "!"},
        "prioritaskan": {"aksen": "#DC2626", "latar": "#FEE2E2", "teks": "#991B1B", "ikon": "!"},
    }[rekom]


def tampilkan_hasil(hasil: dict) -> None:
    if "error" in hasil:
        st.error(f"Maaf, prediksi gagal: {hasil['error']}")
        return

    probabilitas = float(hasil["probabilitas"])
    rekom = hasil["rekomendasi"]
    st.session_state["probabilitas_mentah"] = probabilitas
    gaya = gaya_zona(rekom, tema_aktif())

    st.badge(rekom, color={"aman": "green", "kabari pelanggan": "orange", "prioritaskan": "red"}[rekom])
    st.markdown(
        f"""
        <div style="border-left:6px solid {gaya['aksen']}; background:{gaya['latar']};
                    color:{gaya['teks']}; padding:1rem; border-radius:0.5rem; margin:0.5rem 0;">
          <div style="display:flex; align-items:center; gap:0.5rem; font-weight:600;">
            <span style="color:{gaya['aksen']}; font-size:1.25rem;">{gaya['ikon']}</span>
            <span>{rekom}</span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:2rem;
                      font-weight:700; line-height:1.1; margin-top:0.5rem;">
            {probabilitas:.1%}
          </div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:0.9rem; opacity:0.9;">
            Probabilitas terlambat
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(probabilitas, text=f"{probabilitas:.1%}")
    st.caption(f"Versi model: {hasil['model_version']}")


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
st.caption(
    "Estimasi risiko keterlambatan saat pickup untuk staf layanan pelanggan "
    "dan operasional hub. Model lebih baik untuk meranking paket daripada "
    "memutuskan ya/tidak."
)
url_api = api_url()
st.caption(f"API: {url_api or 'tidak dikonfigurasi (mode lokal)'} | Model: {muat_model_cadangan()[1]['model_version']}")

tab_satu, tab_batch = st.tabs(["Prediksi Satu Paket", "Prediksi Batch CSV"])

with tab_satu:
    with st.container(border=True):
        with st.form("form_paket"):
            kiri, kanan = st.columns(2)
            with kiri:
                st.markdown("**Paket**")
                tanggal = st.date_input("Tanggal pickup", value=date(2025, 6, 15))
                jam = st.slider("Jam pickup", min_value=7, max_value=22, value=9)
                kota = st.selectbox("Kota tujuan", sorted(set(KOTA_ALIAS.values())))
                jarak = st.number_input("Jarak dari Bandung (km)", min_value=0.0, max_value=3000.0, value=150.0)
                transit = st.number_input("Jumlah transit", min_value=0, max_value=10, value=1)

            with kanan:
                st.markdown("**Rute dan layanan**")
                layanan = st.selectbox("Layanan", ["Regular", "Express", "Hemat"])
                janji = st.number_input("Janji tiba (hari)", min_value=1, max_value=14, value=4)
                berat = st.number_input("Berat (kg)", min_value=0.0, max_value=100.0, value=1.2, step=0.1)
                metode = st.selectbox("Metode bayar", ["COD", "Non COD"])
                st.markdown("**Kondisi**")
                cuaca = st.selectbox("Cuaca saat pickup", ["Cerah", "Berawan", "Hujan", "Hujan Lebat"])
                volume_kemarin = st.number_input("Volume hub kemarin", min_value=0, max_value=30000, value=5000)
                volume_7hari = st.number_input("Rata-rata volume hub 7 hari sebelumnya", min_value=0.0, max_value=30000.0, value=5100.0)

            dikirim = st.form_submit_button("Prediksi")

    if dikirim:
        if tanggal is None:
            st.error("Tanggal pickup belum diisi.")
        else:
            payload = {
                "Tanggal Pick up": tanggal.isoformat(),
                "Jam Pick up": int(jam),
                "Kota Tujuan": kota,
                "Jarak Km": float(jarak),
                "Jumlah Transit": int(transit),
                "Layanan": layanan,
                "Janji Hari": int(janji),
                "Berat Kg": float(berat),
                "Metode Bayar": metode,
                "Cuaca Saat Pickup": cuaca,
                "Volume Hub Kemarin": int(volume_kemarin),
                "Volume Hub 7hari": float(volume_7hari),
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
