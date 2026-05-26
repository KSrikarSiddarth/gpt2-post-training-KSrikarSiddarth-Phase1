import argparse
import shutil
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, load_dataset


def load_cfg(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--force", action="store_true", help="Delete cache and re-download")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_cfg = cfg["data"]
    cache_dir = Path(data_cfg["cache_dir"])
    raw_dir = cache_dir / "raw"

    if args.force and cache_dir.exists():
        print(f"Deleting cache at {cache_dir}")
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {data_cfg['dataset_name']}")
    subset = data_cfg.get("subset") or None
    ds = load_dataset(
        data_cfg["dataset_name"],
        subset,
        cache_dir=str(cache_dir),
    )

    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})

    for split_name, split_ds in ds.items():
        size_mb = split_ds.data.nbytes / 1e6
        print(
            f"  {split_name}: {len(split_ds):,} rows | "
            f"columns: {list(split_ds.features)} | ~{size_mb:.1f} MB"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(raw_dir))
    print(f"Saved raw dataset → {raw_dir}")


if __name__ == "__main__":
    main()
