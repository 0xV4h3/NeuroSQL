import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="neurosql/model_weights")
    p.add_argument("--repo_id", default="0xV4h3/neurosql")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    model.push_to_hub(args.repo_id, private=args.private)
    tokenizer.push_to_hub(args.repo_id, private=args.private)

    print(f"[OK] Pushed model + tokenizer to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()