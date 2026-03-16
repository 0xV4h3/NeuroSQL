import os
from transformers.trainer_utils import get_last_checkpoint


def main():
    out_dir = "models/t5_sql_finetuned"
    ckpt = get_last_checkpoint(out_dir)
    if ckpt:
        print(f"[OK] Last checkpoint found: {ckpt}")
    else:
        print("[WARN] No checkpoint found. Start training first.")


if __name__ == "__main__":
    main()