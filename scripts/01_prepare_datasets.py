import os
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "data"
RAW_OUT = os.path.join(OUTPUT_DIR, "unified_sql_dataset.csv")
MANIFEST_OUT = os.path.join(OUTPUT_DIR, "unified_sql_dataset.manifest.json")

DATASETS = [
    ("b-mc2/sql-create-context", {}, "sql-create-context"),
    ("Clinton/Text-to-sql-v1", {}, "text-to-sql-v1"),
    ("gretelai/synthetic_text_to_sql", {}, "synthetic"),
    ("xu3kev/BIRD-SQL-data-train", {}, "bird-sql"),
]

SQL_KEYWORDS_RE = re.compile(r"\b(select|insert|update|delete|with|create)\b", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


@dataclass
class Stats:
    source: str
    total_rows_seen: int = 0
    rows_mapped: int = 0
    rows_kept: int = 0
    rows_dropped_empty: int = 0
    rows_dropped_bad_sql: int = 0
    rows_dropped_no_context: int = 0


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = SPACE_RE.sub(" ", s)
    return s


def canonical_sql_for_dedup(sql: str) -> str:
    x = normalize_text(sql).lower()
    if x.endswith(";"):
        x = x[:-1].strip()
    return x


def canonical_text_for_dedup(text: str) -> str:
    return normalize_text(text).lower()


def looks_like_sql(sql: str) -> bool:
    sql_n = normalize_text(sql)
    if not sql_n:
        return False
    return SQL_KEYWORDS_RE.search(sql_n) is not None


def extract_row(item: Dict[str, Any], source: str) -> Optional[Dict[str, str]]:
    if source == "sql-create-context":
        return {
            "query": item.get("question", ""),
            "context": item.get("context", ""),
            "sql": item.get("answer", ""),
            "source": source,
        }

    if source == "text-to-sql-v1":
        return {
            "query": item.get("input", ""),
            "context": item.get("instruction", "") or item.get("table_schema", "") or "",
            "sql": item.get("response", ""),
            "source": source,
        }

    if source == "synthetic":
        return {
            "query": item.get("sql_prompt", ""),
            "context": item.get("sql_context", ""),
            "sql": item.get("sql", ""),
            "source": source,
        }

    if source == "bird-sql":
        return {
            "query": item.get("question", ""),
            "context": item.get("schema", ""),
            "sql": item.get("SQL", ""),
            "source": source,
        }

    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    stats: Dict[str, Stats] = {src: Stats(source=src) for _, _, src in DATASETS}

    for ds_name, ds_kwargs, ds_source in DATASETS:
        print(f"Downloading {ds_name} ...")
        ds = load_dataset(ds_name, **ds_kwargs)

        for split in ds:
            print(f"  Processing split: {split}")
            for item in ds[split]:
                st = stats[ds_source]
                st.total_rows_seen += 1

                row = extract_row(item, ds_source)
                if row is None:
                    continue
                st.rows_mapped += 1

                row["query"] = normalize_text(row["query"])
                row["context"] = normalize_text(row["context"])
                row["sql"] = normalize_text(row["sql"])

                if not row["query"] or not row["sql"]:
                    st.rows_dropped_empty += 1
                    continue

                if not row["context"]:
                    st.rows_dropped_no_context += 1
                    continue

                if not looks_like_sql(row["sql"]):
                    st.rows_dropped_bad_sql += 1
                    continue

                all_rows.append(row)
                st.rows_kept += 1

    if not all_rows:
        raise RuntimeError("No rows were collected. Check dataset mappings.")

    df = pd.DataFrame(all_rows)

    # Robust dedup by canonical keys
    df["_kq"] = df["query"].map(canonical_text_for_dedup)
    df["_kc"] = df["context"].map(canonical_text_for_dedup)
    df["_ks"] = df["sql"].map(canonical_sql_for_dedup)
    before = len(df)
    df = df.drop_duplicates(subset=["_kq", "_kc", "_ks"]).drop(columns=["_kq", "_kc", "_ks"])
    after = len(df)

    df.to_csv(RAW_OUT, index=False)

    manifest = {
        "output_csv": RAW_OUT,
        "total_rows_before_dedup": before,
        "total_rows_after_dedup": after,
        "dropped_by_dedup": before - after,
        "sources": [asdict(s) for s in stats.values()],
        "sha256_csv": sha256_file(RAW_OUT),
    }

    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved unified dataset: {RAW_OUT}")
    print(f"[OK] Saved manifest: {MANIFEST_OUT}")
    print(f"[INFO] Final rows: {after}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()