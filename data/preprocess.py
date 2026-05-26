import argparse
import json
import re
import shutil
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import yaml
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from transformers import GPT2TokenizerFast


def load_cfg(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Cc" or ch in ("\n", "\t")
    )
    return text


def compute_token_freq(texts, tokenizer, save_path):
    counter = Counter()
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(ids)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(counter, f)
    print(f"Token frequency table saved → {save_path} ({len(counter):,} unique tokens)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_cfg = cfg["data"]
    tokenizer_cfg = cfg["tokenizer"]
    cache_dir = Path(data_cfg["cache_dir"])
    raw_dir = cache_dir / "raw"
    processed_dir = cache_dir / "processed"

    print(f"Loading raw dataset from {raw_dir}")
    ds = load_from_disk(str(raw_dir))

    if isinstance(ds, DatasetDict):
        parts = list(ds.values())
        ds_flat = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    else:
        ds_flat = ds

    print(f"Total rows: {len(ds_flat):,} | columns: {list(ds_flat.features)}")

    text_col = data_cfg["text_column"]
    if text_col not in ds_flat.features:
        raise ValueError(
            f"Column '{text_col}' not found. Available: {list(ds_flat.features)}"
        )

    # Language filter before dropping other columns
    lang_filter = data_cfg.get("language_filter")
    if lang_filter:
        if "language" in ds_flat.features:
            before = len(ds_flat)
            ds_flat = ds_flat.filter(lambda row: row["language"] == lang_filter)
            print(f"Language filter '{lang_filter}': {before:,} → {len(ds_flat):,} rows")
        else:
            warnings.warn("'language' column not found — language_filter skipped.", UserWarning)

    # Keep only text column
    cols_to_remove = [c for c in ds_flat.column_names if c != text_col]
    if cols_to_remove:
        ds_flat = ds_flat.remove_columns(cols_to_remove)

    # Text cleaning
    ds_flat = ds_flat.map(
        lambda row: {text_col: clean_text(row[text_col])},
        desc="Cleaning text",
    )
    ds_flat = ds_flat.filter(lambda row: bool(row[text_col].strip()))
    print(f"After cleaning: {len(ds_flat):,} rows")

    # 90 / 5 / 5 split
    seed = data_cfg["seed"]
    val_ratio = data_cfg["val_ratio"]
    test_ratio = data_cfg["test_ratio"]

    split1 = ds_flat.train_test_split(test_size=val_ratio + test_ratio, seed=seed)
    train_ds = split1["train"]
    relative_test = test_ratio / (val_ratio + test_ratio)
    split2 = split1["test"].train_test_split(test_size=relative_test, seed=seed)
    val_ds = split2["train"]
    test_ds = split2["test"]

    print(
        f"Splits — train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}"
    )

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True)

    for name, split_ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        out = processed_dir / name
        split_ds.save_to_disk(str(out))
        print(f"Saved {name} → {out}")

    print("Computing token frequencies on training split...")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_cfg["name"])
    tokenizer.pad_token = tokenizer.eos_token
    compute_token_freq(train_ds[text_col], tokenizer, cache_dir / "token_freq.json")


if __name__ == "__main__":
    main()
