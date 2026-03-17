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

    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)

    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--save_total_limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataloader_num_workers", type=int, default=None)
    return p.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_config_and_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    final = dict(cfg)
    cli_dict = vars(args)
    for k, v in cli_dict.items():
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
        "max_input_length": 256,
        "max_target_length": 128,
        "num_train_epochs": 3.0,
        "learning_rate": 2e-4,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "grad_accum_steps": 16,
        "save_steps": 200,
        "eval_steps": 400,
        "logging_steps": 20,
        "save_total_limit": 3,
        "seed": 42,
        "dataloader_num_workers": 0,
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


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    p = merge_config_and_args(cfg, args)

    os.makedirs(p["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(p["final_export_dir"]), exist_ok=True)

    dataset = load_from_disk(p["dataset_dir"])
    tokenizer = AutoTokenizer.from_pretrained(p["model_name"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(p["model_name"])
    model.gradient_checkpointing_enable()

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"],
            truncation=True,
            max_length=int(p["max_input_length"]),
        )
        labels = tokenizer(
            text_target=batch["target"],
            truncation=True,
            max_length=int(p["max_target_length"]),
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
    )

    use_fp16 = torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=p["output_dir"],
        num_train_epochs=float(p["num_train_epochs"]),
        learning_rate=float(p["learning_rate"]),
        per_device_train_batch_size=int(p["train_batch_size"]),
        per_device_eval_batch_size=int(p["eval_batch_size"]),
        gradient_accumulation_steps=int(p["grad_accum_steps"]),

        eval_strategy="steps",
        eval_steps=int(p["eval_steps"]),
        save_strategy="steps",
        save_steps=int(p["save_steps"]),
        save_total_limit=int(p["save_total_limit"]),
        logging_steps=int(p["logging_steps"]),

        predict_with_generate=False,
        fp16=use_fp16,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=int(p["dataloader_num_workers"]),

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=int(p["seed"]),
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
    print(f"[INFO] Effective config: {json.dumps(p, ensure_ascii=False, indent=2)}")
    print(f"[INFO] Resume checkpoint: {resume_checkpoint}")

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Saving safe checkpoint...")
        trainer.save_model()
        trainer.save_state()
        raise
    finally:
        print("[INFO] Exporting latest model to final_export_dir...")
        trainer.save_model(p["final_export_dir"])
        tokenizer.save_pretrained(p["final_export_dir"])

        info = {
            "effective_config": p,
            "resolved_resume_checkpoint": resume_checkpoint,
        }
        with open(os.path.join(p["output_dir"], "train_run_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[OK] Training finished. Final model: {p['final_export_dir']}")


if __name__ == "__main__":
    main()