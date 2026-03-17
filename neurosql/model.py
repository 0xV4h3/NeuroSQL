from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .utils import build_prompt, postprocess_sql


@dataclass
class GenerateConfig:
    max_new_tokens: int = 128
    num_beams: int = 4
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


class NeuroSQLModel:
    """
    NeuroSQL inference wrapper.
    Supports loading from:
      - HF Hub repo id: "0xV4h3/neurosql"
      - local path: "neurosql/model_weights"
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "0xV4h3/neurosql",
        device: Optional[str] = None,
    ) -> "NeuroSQLModel":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False

        return cls(model=model, tokenizer=tokenizer, device=device)

    def generate(
        self,
        query: str,
        context: str,
        *,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        prompt = build_prompt(query=query, context=context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                early_stopping=True,
            )

        sql = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return postprocess_sql(sql)

    @classmethod
    def from_local_weights_or_hub(
        cls,
        local_dir: str = "neurosql/model_weights",
        hub_repo_id: str = "0xV4h3/neurosql",
        device: Optional[str] = None,
    ) -> "NeuroSQLModel":
        if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
            return cls.from_pretrained(local_dir, device=device)
        return cls.from_pretrained(hub_repo_id, device=device)