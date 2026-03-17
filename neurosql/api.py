from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .model import NeuroSQLModel

app = FastAPI(title="NeuroSQL API", version="0.1.0")

_MODEL: Optional[NeuroSQLModel] = None


class GenerateRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    context: str = Field(..., description="SQL schema context")
    max_new_tokens: int = 128
    num_beams: int = 4
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


class GenerateResponse(BaseModel):
    sql: str


def get_model() -> NeuroSQLModel:
    global _MODEL
    if _MODEL is None:
        local_dir = "neurosql/model_weights"
        hub_id = os.getenv("NEUROSQL_HF_REPO", "0xV4h3/neurosql")
        _MODEL = NeuroSQLModel.from_local_weights_or_hub(local_dir=local_dir, hub_repo_id=hub_id)
    return _MODEL


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model = get_model()
    sql = model.generate(
        query=req.query,
        context=req.context,
        max_new_tokens=req.max_new_tokens,
        num_beams=req.num_beams,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return GenerateResponse(sql=sql)