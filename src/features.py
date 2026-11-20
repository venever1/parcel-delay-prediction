"""Fungsi pembersihan data yang dipakai saat pelatihan dan prediksi.

Aturan penting: fungsi di sini hanya menangani masalah yang pasti (duplikat,
label, nilai mustahil). Imputasi tetap dilakukan setelah split, memakai
Pipeline yang di-fit pada train saja.
"""

from __future__ import annotations

import pandas as pd

# Batas wajar paket kurir. Nilai di atas ini kemungkinan salah satuan (gram)
# atau salah input, jadi tidak dikoreksi agar tidak menebak.
MAX_BERAT_KG = 100.0

# Batas ekor distribusi volume hub. Nilai di atas ini ditandai, bukan dibuang,
# karena bisa jadi lonjakan nyata.
VOLUME_HUB_EKSTREM = 9000

# Rentang SLA yang lazim untuk tiap layanan, dipakai hanya untuk menandai.
JANJI_VALID = {
    "EXPRESS": (1, 3),
    "REG": (2, 4),
    "HEMAT": (4, 6),
}

# Alias ejaan yang tidak ambigu. Kunci memakai huruf kecil.
KOTA_ALIAS = {
    "jakarta": "Jakarta",
    "jkt": "Jakarta",
    "dki jakarta": "Jakarta",
    "bekasi": "Bekasi",
    "kota bekasi": "Bekasi",
    "surabaya": "Surabaya",
    "sby": "Surabaya",
    "medan": "Medan",
    "yogyakarta": "Yogyakarta",
    "di yogyakarta": "Yogyakarta",
    "jogja": "Yogyakarta",
    "yogya": "Yogyakarta",
    "denpasar": "Denpasar",
    # Di data ini jarak "Bali" sama dengan Denpasar dan Denpasar satu-satunya
    # kota Bali yang muncul; pemetaan ini asumsi yang dicatat di laporan.
    "bali": "Denpasar",
    "bogor": "Bogor",
    "semarang": "Semarang",
    "tasikmalaya": "Tasikmalaya",
    "cirebon": "Cirebon",
    "malang": "Malang",
    "makassar": "Makassar",
    "palembang": "Palembang",
    "balikpapan": "Balikpapan",
    "pontianak": "Pontianak",
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data mentah tanpa melakukan imputasi."""
    data = df.copy()

    # Duplikat identik tidak menambah informasi dan bisa membobot ganda.
    data = data.drop_duplicates().reset_index(drop=True)

    # Rapikan teks sebelum dipetakan agar perbedaan spasi/kapital tidak lolos.
    for col in ["kota_tujuan", "layanan", "metode_bayar", "cuaca_saat_pickup"]:
        data[col] = data[col].astype("string").str.strip()

    layanan_mentah = data["layanan"].str.upper()
    data["flag_layanan_alias"] = layanan_mentah.ne(data["layanan"])
    data["layanan"] = layanan_mentah.replace({"EXP": "EXPRESS"})

    # Simpan versi mentah untuk menandai kota yang ejaannya diubah.
    kota_mentah = data["kota_tujuan"]
    kota_kunci = kota_mentah.str.lower()
    data["kota_tujuan"] = kota_kunci.map(KOTA_ALIAS).fillna(kota_mentah.str.title())
    data["flag_kota_alias"] = data["kota_tujuan"].ne(kota_mentah)

    metode = data["metode_bayar"].str.lower()
    data["metode_bayar"] = metode.map({"cod": "COD", "non_cod": "non_COD"}).fillna(
        data["metode_bayar"]
    )
    data["cuaca_saat_pickup"] = data["cuaca_saat_pickup"].str.lower()

    # Tanggal tidak valid atau di luar periode data tetap disimpan, hanya ditandai.
    tanggal = pd.to_datetime(data["tanggal_pickup"], errors="coerce")
    data["tanggal_pickup"] = tanggal
    data["flag_tanggal_invalid"] = tanggal.isna() | ~tanggal.between(
        "2025-01-01", "2026-06-30"
    )

    for col in ["jarak_km", "jumlah_transit", "janji_hari", "berat_kg", "volume_hub_hari"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Nilai mustahil dijadikan NaN; imputasi menunggu setelah split.
    data.loc[data["jarak_km"] < 0, "jarak_km"] = pd.NA
    data.loc[data["jumlah_transit"] < 0, "jumlah_transit"] = pd.NA
    data.loc[data["janji_hari"] <= 0, "janji_hari"] = pd.NA
    data.loc[data["volume_hub_hari"] <= 0, "volume_hub_hari"] = pd.NA

    jam = pd.to_numeric(data["jam_pickup"], errors="coerce")
    data["jam_pickup"] = jam.where(jam.between(7, 22))
    data["flag_jam_invalid"] = data["jam_pickup"].isna()

    berat_invalid = data["berat_kg"].isna() | (data["berat_kg"] <= 0) | (
        data["berat_kg"] > MAX_BERAT_KG
    )
    data["flag_berat_invalid"] = berat_invalid
    data.loc[berat_invalid, "berat_kg"] = pd.NA

    # Masalah yang belum bisa diputuskan hanya ditandai, tidak diubah.
    data["flag_jarak_missing"] = data["jarak_km"].isna()
    data["flag_cuaca_missing"] = data["cuaca_saat_pickup"].isna()
    data["flag_volume_ekstrem"] = data["volume_hub_hari"] > VOLUME_HUB_EKSTREM

    janji_min = data["layanan"].map({k: v[0] for k, v in JANJI_VALID.items()})
    janji_max = data["layanan"].map({k: v[1] for k, v in JANJI_VALID.items()})
    data["flag_janji_tidak_sesuai"] = (
        data["janji_hari"].notna()
        & (
            (data["janji_hari"] < janji_min)
            | (data["janji_hari"] > janji_max)
        )
    ).fillna(False).astype(bool)

    return data
