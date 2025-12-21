from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import time

from eda_cli.core import (
    summarize_dataset,
    missing_table,
    compute_quality_flags,
)

app = FastAPI(
    title="EDA Dataset Quality Service",
    version="0.1.0",
    description="HTTP service for dataset quality assessment based on eda-cli",
)


#models

class QualityRequest(BaseModel):
    n_rows: int
    n_cols: int
    max_missing_share: float
    numeric_cols: int
    categorical_cols: int


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    flags: dict
    latency_ms: float
    dataset_shape: tuple


#basic enpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "eda-cli-api",
        "version": "0.1.0",
    }


@app.post("/quality", response_model=QualityResponse)
def quality(req: QualityRequest):
    start = time.perf_counter()

    flags = {
        "too_few_rows": req.n_rows < 100,
        "too_many_missing": req.max_missing_share > 0.3,
    }

    quality_score = 1.0
    for v in flags.values():
        if v:
            quality_score -= 0.2
    quality_score = max(0.0, quality_score)

    ok_for_model = quality_score >= 0.5

    latency_ms = (time.perf_counter() - start) * 1000

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=quality_score,
        flags=flags,
        latency_ms=latency_ms,
        dataset_shape=(req.n_rows, req.n_cols),
    )


@app.post("/quality-from-csv", response_model=QualityResponse)
def quality_from_csv(file: UploadFile = File(...)):
    start = time.perf_counter()

    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if df.empty:
        raise HTTPException(status_code=400, detail="Empty CSV file")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    quality_score = flags.get("quality_score", 0.0)
    ok_for_model = quality_score >= 0.5

    latency_ms = (time.perf_counter() - start) * 1000

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=quality_score,
        flags=flags,
        latency_ms=latency_ms,
        dataset_shape=df.shape,
    )


#custom endpoint
@app.post("/quality-flags-from-csv")
def quality_flags_from_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if df.empty:
        raise HTTPException(status_code=400, detail="Empty CSV file")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    return {
        "flags": flags
    }