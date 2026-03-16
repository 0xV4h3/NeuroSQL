import os
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from_dir", default="models/t5_sql_finetuned")
    p.add_argument("--to_dir", default="neurosql/model_weights")
    args = p.parse_args()

    os.makedirs(args.to_dir, exist_ok=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.from_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.from_dir)

    model.save_pretrained(args.to_dir)
    tokenizer.save_pretrained(args.to_dir)

    print(f"[OK] Exported model to {args.to_dir}")


if __name__ == "__main__":
    main()