"""Cek cepat: jalur fitur training dan API harus sama untuk satu baris."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (
    FEATURE_COLUMNS_V1,
    KotaTargetEncoder,
    build_features,
    siapkan_fitur,
)

BARIS = {
    "tanggal_pickup": "2025-06-15",
    "jam_pickup": 14,
    "kota_tujuan": "Jkt",
    "jarak_km": 150.0,
    "jumlah_transit": 2,
    "layanan": "reg",
    "janji_hari": 3,
    "berat_kg": 2.5,
    "metode_bayar": "non_COD",
    "cuaca_saat_pickup": None,
    "volume_hub_hari": 5000,
    "volume_hub_kemarin": 4900,
    "volume_hub_7hari": 5050.0,
}

data = pd.DataFrame([BARIS])
metadata = {
    "feature_builder": "build_features_v1",
    "fitur_masukan": FEATURE_COLUMNS_V1,
}

latih = build_features(data)
api = siapkan_fitur(data, metadata)
pd.testing.assert_frame_equal(latih, api)

X_latih = pd.DataFrame(
    {
        "kota_tujuan": ["Jakarta", "Jakarta", "Bekasi", "Bekasi"],
        "jarak_km": [1.0, 2.0, 3.0, 4.0],
    }
)
encoder = KotaTargetEncoder(smooth=1.0).fit(X_latih, [0, 1, 1, 0])
X_baru = pd.DataFrame(
    {"kota_tujuan": ["Jakarta", "KotaBaru"], "jarak_km": [5.0, 6.0]}
)
hasil = encoder.transform(X_baru)
assert pd.notna(hasil.loc[0, "kota_target"])
assert pd.isna(hasil.loc[1, "kota_target"])

print(f"OK: training dan API sama untuk {len(FEATURE_COLUMNS_V1)} fitur")
print("OK: kota baru menjadi NaN pada target encoder")
print(f"Fitur: {', '.join(FEATURE_COLUMNS_V1)}")
