import os
import argparse
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint


def has_model_config(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def resolve_source_dir(from_dir: str) -> str:
    if has_model_config(from_dir):
        return from_dir

    ckpt = get_last_checkpoint(from_dir)
    if ckpt and has_model_config(ckpt):
        return ckpt

    p = Path(from_dir)
    if p.exists() and p.is_dir():
        checkpoints = sorted(
            [x for x in p.iterdir() if x.is_dir() and x.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else -1,
        )
        if checkpoints:
            last = str(checkpoints[-1])
            if has_model_config(last):
                return last

    raise FileNotFoundError(
        f"No valid model config found in '{from_dir}' or its checkpoints.\n"
        f"Run training first or pass --from_dir <valid_checkpoint_or_model_dir>."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from_dir", default="models/t5_sql_finetuned")
    p.add_argument("--to_dir", default="neurosql/model_weights")
    args = p.parse_args()

    os.makedirs(args.to_dir, exist_ok=True)

    src = resolve_source_dir(args.from_dir)
    print(f"[INFO] Export source: {src}")

    model = AutoModelForSeq2SeqLM.from_pretrained(src)
    tokenizer = AutoTokenizer.from_pretrained(src)

    model.save_pretrained(args.to_dir)
    tokenizer.save_pretrained(args.to_dir)

    print(f"[OK] Exported model to {args.to_dir}")


if __name__ == "__main__":
    main()