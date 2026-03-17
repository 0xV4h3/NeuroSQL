import os
import json
import argparse
from typing import Optional, Dict, Any

import torch
import yaml
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_4gb.yaml", help="Path to YAML config")

    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--final_export_dir", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--resume", default=None, help="auto | none | /path/to/checkpoint")

    p.add_argument("--max_input_length", type=int, default=None)
    p.add_argument("--max_target_length", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)

    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)

    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--save_total_limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataloader_num_workers", type=int, default=None)

    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    return p.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config_and_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    final = dict(cfg)
    cli = vars(args)
    for k, v in cli.items():
        if k == "config":
            continue
        if v is not None:
            final[k] = v

    defaults = {
        "dataset_dir": "data/t5_sql_dataset",
        "output_dir": "models/t5_sql_finetuned",
        "final_export_dir": "neurosql/model_weights",
        "model_name": "google/flan-t5-small",
        "resume": "auto",

        "max_input_length": 192,
        "max_target_length": 128,
        "num_train_epochs": 1.0,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,

        "train_batch_size": 4,
        "eval_batch_size": 1,
        "grad_accum_steps": 4,

        "save_steps": 400,
        "eval_steps": 400,
        "logging_steps": 20,
        "save_total_limit": 3,

        "seed": 42,
        "dataloader_num_workers": 0,

        "max_train_samples": 80000,
        "max_eval_samples": 4000,
    }
    for k, v in defaults.items():
        final.setdefault(k, v)

    return final


def resolve_resume(output_dir: str, resume_arg: str) -> Optional[str]:
    if resume_arg == "none":
        return None
    if resume_arg == "auto":
        return get_last_checkpoint(output_dir)
    return resume_arg


def sanitize_numeric_config(p: Dict[str, Any]) -> Dict[str, Any]:
    p["learning_rate"] = float(p["learning_rate"])
    p["num_train_epochs"] = float(p["num_train_epochs"])
    p["warmup_ratio"] = float(p["warmup_ratio"])
    p["max_grad_norm"] = float(p["max_grad_norm"])

    int_keys = [
        "max_input_length", "max_target_length",
        "train_batch_size", "eval_batch_size", "grad_accum_steps",
        "save_steps", "eval_steps", "logging_steps", "save_total_limit",
        "seed", "dataloader_num_workers", "max_train_samples", "max_eval_samples"
    ]
    for k in int_keys:
        if p.get(k) is not None:
            p[k] = int(p[k])
    return p


def validate_config(p: Dict[str, Any]):
    if p["save_steps"] % p["eval_steps"] != 0:
        raise ValueError(
            f"save_steps ({p['save_steps']}) must be a multiple of eval_steps ({p['eval_steps']}) "
            f"when load_best_model_at_end=True"
        )
    if p["train_batch_size"] < 1 or p["eval_batch_size"] < 1:
        raise ValueError("train_batch_size and eval_batch_size must be >= 1")
    if p["grad_accum_steps"] < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if p["max_input_length"] <= 0 or p["max_target_length"] <= 0:
        raise ValueError("max_input_length and max_target_length must be > 0")


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    p = merge_config_and_args(cfg, args)
    p = sanitize_numeric_config(p)
    validate_config(p)

    os.makedirs(p["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(p["final_export_dir"]), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    dataset = load_from_disk(p["dataset_dir"])
    tokenizer = AutoTokenizer.from_pretrained(p["model_name"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(p["model_name"])
    model.gradient_checkpointing_enable()

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"],
            truncation=True,
            max_length=p["max_input_length"],
        )
        labels = tokenizer(
            text_target=batch["target"],
            truncation=True,
            max_length=p["max_target_length"],
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    if p["max_train_samples"] is not None:
        n = min(len(tokenized["train"]), p["max_train_samples"])
        tokenized["train"] = tokenized["train"].select(range(n))
    if p["max_eval_samples"] is not None:
        n = min(len(tokenized["validation"]), p["max_eval_samples"])
        tokenized["validation"] = tokenized["validation"].select(range(n))

    print(f"[INFO] Train samples used: {len(tokenized['train'])}")
    print(f"[INFO] Eval samples used: {len(tokenized['validation'])}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=p["output_dir"],
        num_train_epochs=p["num_train_epochs"],
        learning_rate=p["learning_rate"],
        warmup_ratio=p["warmup_ratio"],
        max_grad_norm=p["max_grad_norm"],

        per_device_train_batch_size=p["train_batch_size"],
        per_device_eval_batch_size=p["eval_batch_size"],
        gradient_accumulation_steps=p["grad_accum_steps"],

        eval_strategy="steps",
        eval_steps=p["eval_steps"],
        save_strategy="steps",
        save_steps=p["save_steps"],
        save_total_limit=p["save_total_limit"],
        logging_steps=p["logging_steps"],

        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=p["dataloader_num_workers"],

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=p["seed"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    resume_checkpoint = resolve_resume(p["output_dir"], str(p["resume"]))
    print(f"[INFO] Effective config:\n{json.dumps(p, ensure_ascii=False, indent=2)}")
    print(f"[INFO] Resume checkpoint: {resume_checkpoint}")

    interrupted = False
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[WARN] Interrupted by user. Saving safe checkpoint/state...")
        trainer.save_model()
        trainer.save_state()

    print("[INFO] Exporting latest model to final_export_dir...")
    trainer.save_model(p["final_export_dir"])
    tokenizer.save_pretrained(p["final_export_dir"])

    info = {
        "effective_config": p,
        "resolved_resume_checkpoint": resume_checkpoint,
        "interrupted": interrupted,
    }
    with open(os.path.join(p["output_dir"], "train_run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    if interrupted:
        print("[OK] Interrupted safely. You can resume with: make resume")
    else:
        print(f"[OK] Training finished. Final model: {p['final_export_dir']}")


if __name__ == "__main__":
    main()