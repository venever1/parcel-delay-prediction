"""Smoke test API: 3 masukan normal dan 5 masukan tidak wajar."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.api.main import app

BASE = {
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

NORMAL = {
    "aman": {},
    "kabari": {
        "kota_tujuan": "Surabaya",
        "jarak_km": 700.0,
        "jumlah_transit": 2,
        "janji_hari": 3,
        "berat_kg": 4.5,
        "metode_bayar": "COD",
        "cuaca_saat_pickup": "hujan",
        "volume_hub_hari": 6000,
        "jam_pickup": 15,
    },
    "prioritaskan": {
        "kota_tujuan": "Medan",
        "jarak_km": 1800.0,
        "jumlah_transit": 5,
        "layanan": "EXPRESS",
        "janji_hari": 1,
        "berat_kg": 2.0,
        "metode_bayar": "COD",
        "cuaca_saat_pickup": "hujan_lebat",
        "volume_hub_hari": 12000,
        "jam_pickup": 20,
    },
}

ABNORMAL = {
    "jam 25": {"jam_pickup": 25},
    "berat 0": {"berat_kg": 0},
    "jarak negatif": {"jarak_km": -5},
    "layanan KILAT": {"layanan": "KILAT"},
    "kota kosong": {"kota_tujuan": "   "},
}


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        print("HEALTH", health.status_code, health.json())

        print("\nNORMAL")
        for nama, ubah in NORMAL.items():
            respons = client.post("/predict", json={**BASE, **ubah})
            data = respons.json()
            print(
                f"{nama:14s} status={respons.status_code} "
                f"probabilitas={data.get('probabilitas', 0):.4f} "
                f"rekomendasi={data.get('rekomendasi')} "
                f"versi={data.get('model_version')}"
            )

        print("\nABNORMAL")
        for nama, ubah in ABNORMAL.items():
            respons = client.post("/predict", json={**BASE, **ubah})
            data = respons.json()
            pesan = data.get("errors", [{}])[0].get("message", data.get("detail"))
            print(f"{nama:14s} status={respons.status_code} pesan={pesan}")


if __name__ == "__main__":
    main()
