"""API prediksi keterlambatan paket.

Model dan metadata dimuat sekali saat startup. Semua validasi masukan dilakukan
Pydantic; data yang lolos dibersihkan dengan fungsi yang sama seperti pelatihan.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features import KOTA_ALIAS, siapkan_fitur

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"


class PackageInput(BaseModel):
    """Data satu paket saat pickup."""

    model_config = ConfigDict(extra="forbid")

    tanggal_pickup: date
    jam_pickup: int = Field(ge=7, le=22)
    kota_tujuan: str = Field(min_length=1)
    jarak_km: float = Field(ge=0, allow_inf_nan=False)
    jumlah_transit: int = Field(ge=0)
    layanan: Literal["EXPRESS", "REG", "HEMAT"]
    janji_hari: int = Field(ge=1)
    berat_kg: float = Field(gt=0, le=100, allow_inf_nan=False)
    metode_bayar: Literal["COD", "non_COD"]
    cuaca_saat_pickup: Literal["cerah", "berawan", "hujan", "hujan_lebat"]
    volume_hub_hari: int = Field(gt=0)
    volume_hub_kemarin: int = Field(gt=0)
    volume_hub_7hari: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("kota_tujuan")
    @classmethod
    def cek_kota(cls, nilai: str) -> str:
        nilai = nilai.strip()
        if not nilai:
            raise ValueError("kota_tujuan tidak boleh kosong")
        if nilai.lower() not in KOTA_ALIAS:
            raise ValueError("kota_tujuan tidak dikenal; gunakan kota yang ada di data pelatihan")
        return nilai


class PredictionResponse(BaseModel):
    probabilitas: float
    rekomendasi: Literal["aman", "kabari pelanggan", "prioritaskan"]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dimuat sekali agar setiap request tidak membaca disk.
    app.state.model = joblib.load(MODEL_PATH)
    app.state.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    yield


app = FastAPI(
    title="Prediksi Keterlambatan Paket",
    description="Baseline regresi logistik untuk data paket saat pickup.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validasi_gagal(request: Request, exc: RequestValidationError):
    # Ringkas error Pydantic agar pesan mudah dibaca pengguna API.
    errors = []
    for err in exc.errors():
        lokasi = [str(bagian) for bagian in err["loc"] if bagian != "body"]
        errors.append({"field": ".".join(lokasi), "message": err["msg"]})
    return JSONResponse(
        status_code=422,
        content={"detail": "Input tidak valid. Periksa kembali data paket.", "errors": errors},
    )


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_version": request.app.state.metadata["model_version"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PackageInput, request: Request) -> PredictionResponse:
    model = request.app.state.model
    metadata = request.app.state.metadata

    try:
        # Jalur fitur yang sama dengan pelatihan, dipilih dari metadata model.
        data = pd.DataFrame([payload.model_dump(mode="json")])
        fitur = siapkan_fitur(data, metadata)
        probabilitas = float(model.predict_proba(fitur)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Data paket tidak dapat diproses. Periksa kembali nilai dan kategori.",
        ) from exc

    thresholds = metadata["thresholds"]
    if probabilitas >= float(thresholds["prioritaskan"]):
        rekomendasi = "prioritaskan"
    elif probabilitas >= float(thresholds["kabari_pelanggan"]):
        rekomendasi = "kabari pelanggan"
    else:
        rekomendasi = "aman"

    return PredictionResponse(
        probabilitas=probabilitas,
        rekomendasi=rekomendasi,
        model_version=metadata["model_version"],
    )
