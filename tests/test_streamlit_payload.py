"""Regresi: payload form default Streamlit harus lolos validasi PackageInput."""

from __future__ import annotations

import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

os.environ.pop("API_URL", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

import app.ui.streamlit_app as ui
from app.api.main import PackageInput

APP = ROOT / "app" / "ui" / "streamlit_app.py"

KOLOM_ASLI = [
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
    "volume_hub_kemarin",
    "volume_hub_7hari",
]

CONTOH = {
    "tanggal_pickup": "2025-06-15",
    "jam_pickup": 9,
    "kota_tujuan": "Jakarta",
    "jarak_km": 150.0,
    "jumlah_transit": 1,
    "layanan": "REG",
    "janji_hari": 4,
    "berat_kg": 1.2,
    "metode_bayar": "non_COD",
    "cuaca_saat_pickup": "cerah",
    "volume_hub_kemarin": 5000,
    "volume_hub_7hari": 5100.0,
}


def cek_payload_default() -> dict:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    at.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.error, [elemen.value for elemen in at.error]

    payload = at.session_state["payload_terakhir"]
    assert list(payload) == KOLOM_ASLI
    PackageInput.model_validate(payload)
    assert payload["layanan"] in {"REG", "EXPRESS", "HEMAT"}
    assert payload["metode_bayar"] in {"COD", "non_COD"}
    assert payload["cuaca_saat_pickup"] in {"cerah", "berawan", "hujan", "hujan_lebat"}
    assert at.session_state["probabilitas_mentah"] >= 0
    return payload


def cek_batch() -> None:
    keluaran = ui.prediksi_batch(pd.DataFrame([CONTOH]))
    assert list(keluaran.columns[: len(KOLOM_ASLI)]) == KOLOM_ASLI
    assert pd.notna(keluaran.loc[0, "probabilitas"])
    assert keluaran.loc[0, "error"] == ""

    unduhan = pd.read_csv(StringIO(keluaran.to_csv(index=False)))
    assert list(unduhan.columns[: len(KOLOM_ASLI)]) == KOLOM_ASLI


if __name__ == "__main__":
    payload = cek_payload_default()
    cek_batch()
    print("OK: payload default lolos PackageInput")
    print(
        "OK: kategori sah "
        f"{payload['layanan']}/{payload['metode_bayar']}/{payload['cuaca_saat_pickup']}"
    )
    print("OK: batch memakai kolom asli dan unduhan mempertahankannya")
