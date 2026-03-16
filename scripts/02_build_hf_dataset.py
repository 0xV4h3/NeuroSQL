import os
import json
import random
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict


def make_prompt(context: str, query: str) -> str:
    return f"translate to SQL: context: {context} question: {query}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/unified_sql_dataset.csv")
    parser.add_argument("--output_dir", default="data/t5_sql_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.94)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    df = pd.read_csv(args.input_csv)

    # Safety cleanup
    for c in ["query", "context", "sql", "source"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = df.dropna(subset=["query", "context", "sql"])
    df["query"] = df["query"].astype(str).str.strip()
    df["context"] = df["context"].astype(str).str.strip()
    df["sql"] = df["sql"].astype(str).str.strip()
    df = df[(df["query"] != "") & (df["context"] != "") & (df["sql"] != "")]

    rows = df.to_dict("records")
    random.Random(args.seed).shuffle(rows)

    n = len(rows)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    def to_hf_dict(items):
        return {
            "input": [make_prompt(x["context"], x["query"]) for x in items],
            "target": [x["sql"] for x in items],
            "query": [x["query"] for x in items],
            "context": [x["context"] for x in items],
            "source": [x["source"] for x in items],
        }

    ds = DatasetDict({
        "train": Dataset.from_dict(to_hf_dict(train_rows)),
        "validation": Dataset.from_dict(to_hf_dict(val_rows)),
        "test": Dataset.from_dict(to_hf_dict(test_rows)),
    })

    ds.save_to_disk(args.output_dir)

    stats = {
        "total": n,
        "train": n_train,
        "validation": n_val,
        "test": n_test,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved dataset to: {args.output_dir}")
    print(f"[INFO] Split stats: {stats}")


if __name__ == "__main__":
    main()