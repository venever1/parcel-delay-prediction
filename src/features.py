"""Fungsi pembersihan data yang dipakai saat pelatihan dan prediksi.

Aturan penting: fungsi di sini hanya menangani masalah yang pasti (duplikat,
label, nilai mustahil). Imputasi tetap dilakukan setelah split, memakai
Pipeline yang di-fit pada train saja.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

FEATURE_NUMERIC_V1 = [
    "jam_pickup",
    "jarak_km",
    "jumlah_transit",
    "janji_hari",
    "berat_kg",
    "hari_dalam_minggu",
    "is_weekend",
    "hari_dalam_bulan",
    "is_akhir_bulan",
    "jarak_per_janji",
    "jarak_per_transit",
    "flag_jarak_missing",
    "flag_cuaca_missing",
]

FEATURE_CATEGORICAL_V1 = [
    "jam_bucket",
    "kota_tujuan",
    "layanan",
    "metode_bayar",
    "cuaca_saat_pickup",
]

FEATURE_COLUMNS_V1 = FEATURE_NUMERIC_V1 + FEATURE_CATEGORICAL_V1


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

    # Sklearn lebih stabil dengan np.nan daripada pd.NA pada kolom teks.
    for kolom in ["kota_tujuan", "layanan", "metode_bayar", "cuaca_saat_pickup"]:
        data[kolom] = data[kolom].astype(object).where(data[kolom].notna(), np.nan)

    return data


class KotaCategoryEncoder(BaseEstimator, TransformerMixin):
    """Ubah kota menjadi kategori yang dikenal LightGBM.

    Kategori diambil hanya dari train. Kota baru menjadi NaN, bukan-error.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "KotaCategoryEncoder":
        self.categories_ = sorted(str(v) for v in X["kota_tujuan"].dropna().unique())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        hasil = X.copy()
        hasil["kota_tujuan"] = pd.Categorical(
            hasil["kota_tujuan"], categories=self.categories_
        )
        return hasil

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else ["kota_tujuan"])


class KotaTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding kota yang di-fit hanya pada train.

    Kota yang tidak dikenal menghasilkan NaN, jadi aman untuk prediksi baru.
    """

    def __init__(self, smooth: float = 10.0):
        self.smooth = smooth

    def fit(self, X: pd.DataFrame, y) -> "KotaTargetEncoder":
        target = pd.Series(list(y), index=X.index, dtype=float)
        pasangan = pd.DataFrame(
            {"kota": X["kota_tujuan"].astype("string"), "target": target}
        ).dropna()
        stat = pasangan.groupby("kota")["target"].agg(["mean", "count"])
        prior = float(target.mean())
        self.prior_ = prior
        self.mapping_ = (
            (stat["mean"] * stat["count"] + prior * self.smooth)
            / (stat["count"] + self.smooth)
        ).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        hasil = X.copy()
        kode = hasil["kota_tujuan"].astype("string").map(self.mapping_)
        hasil["kota_target"] = pd.Series(kode, index=hasil.index).astype(float)
        return hasil.drop(columns=["kota_tujuan"])

    def get_feature_names_out(self, input_features=None):
        nama = [n for n in (input_features or []) if n != "kota_tujuan"]
        return np.asarray(nama + ["kota_target"], dtype=object)


def make_logistic_preprocessor() -> ColumnTransformer:
    """Preprocessing untuk regresi logistik: median/modus, scaling, one-hot."""
    numerik = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    kategorikal = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numerik", numerik, FEATURE_NUMERIC_V1),
            ("kategorikal", kategorikal, FEATURE_CATEGORICAL_V1),
        ],
        remainder="drop",
    )


def make_tree_preprocessor() -> ColumnTransformer:
    """Preprocessing untuk model tree dengan target encoding kota."""
    kategorikal_lain = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("kota", KotaTargetEncoder(), ["kota_tujuan"]),
            (
                "kategorikal_lain",
                kategorikal_lain,
                [kolom for kolom in FEATURE_CATEGORICAL_V1 if kolom != "kota_tujuan"],
            ),
            ("numerik", "passthrough", FEATURE_NUMERIC_V1),
        ],
        remainder="drop",
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Buat fitur dari data satu paket saat pickup.

    Fitur yang dibuat hanya memakai informasi yang sudah diketahui saat pickup.
    `volume_hub_hari` **sengaja tidak dipakai**: nilai itu adalah total paket
    sehari penuh dan bisa baru lengkap setelah pickup, sehingga berisiko
    membocorkan informasi masa depan. Jika nanti tersedia volume berjalan
    sampai jam pickup, fitur itu dapat ditambahkan dengan versi baru.
    """
    data = clean_raw(df)
    tanggal = data["tanggal_pickup"]
    jam = data["jam_pickup"]
    jarak = data["jarak_km"]
    janji = data["janji_hari"]
    transit = data["jumlah_transit"]

    hari = tanggal.dt.dayofweek
    fitur = pd.DataFrame(index=data.index)

    fitur["jam_pickup"] = jam
    fitur["jarak_km"] = jarak
    fitur["jumlah_transit"] = transit
    fitur["janji_hari"] = janji
    fitur["berat_kg"] = data["berat_kg"]

    fitur["hari_dalam_minggu"] = hari
    fitur["is_weekend"] = (hari >= 5).astype("int8")
    fitur["hari_dalam_bulan"] = tanggal.dt.day
    fitur["is_akhir_bulan"] = (
        tanggal == (tanggal + pd.offsets.MonthEnd(0))
    ).astype("int8")

    fitur["jam_bucket"] = pd.cut(
        jam,
        bins=[6, 9, 12, 15, 18, 22],
        labels=["pagi", "siang", "sore", "petang", "malam"],
    ).astype("string")

    # Pembagian dijaga tetap aman: penyebab nol/negatif menjadi NaN, bukan inf.
    fitur["jarak_per_janji"] = jarak / janji.where(janji > 0)
    fitur["jarak_per_transit"] = jarak / transit.where(transit > 0)

    fitur["flag_jarak_missing"] = jarak.isna()
    fitur["flag_cuaca_missing"] = data["cuaca_saat_pickup"].isna()

    fitur["kota_tujuan"] = data["kota_tujuan"]
    fitur["layanan"] = data["layanan"]
    fitur["metode_bayar"] = data["metode_bayar"]
    fitur["cuaca_saat_pickup"] = data["cuaca_saat_pickup"]

    # SimpleImputer lebih stabil dengan np.nan daripada pd.NA pada kolom teks.
    for kolom in FEATURE_CATEGORICAL_V1:
        fitur[kolom] = fitur[kolom].astype(object).where(fitur[kolom].notna(), np.nan)

    return fitur[FEATURE_COLUMNS_V1]


def siapkan_fitur(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Jalur fitur yang dipakai sama oleh training, API, dan Streamlit.

    Model lama tanpa `feature_builder` tetap memakai fitur asli. Model baru
    dengan `feature_builder = build_features_v1` memakai fitur turunan di atas.
    """
    if metadata.get("feature_builder") == "build_features_v1":
        return build_features(df)[metadata["fitur_masukan"]]
    return clean_raw(df)[metadata["fitur_masukan"]]
