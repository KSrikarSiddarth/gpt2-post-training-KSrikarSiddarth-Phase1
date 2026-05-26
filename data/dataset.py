from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast


class TextChunkDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, stride):
        self.chunks = []
        for text in texts:
            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_overflowing_tokens=True,
                max_length=max_length,
                stride=stride,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            for i in range(len(enc["input_ids"])):
                input_ids = torch.tensor(enc["input_ids"][i], dtype=torch.long)
                attention_mask = torch.tensor(enc["attention_mask"][i], dtype=torch.long)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                self.chunks.append(
                    {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def get_dataloader(split, cfg, tokenizer=None):
    data_cfg = cfg["data"]
    tokenizer_cfg = cfg["tokenizer"]
    cache_dir = Path(data_cfg["cache_dir"])

    ds = load_from_disk(str(cache_dir / "processed" / split))
    texts = ds[data_cfg["text_column"]]

    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_cfg["name"])
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TextChunkDataset(
        texts,
        tokenizer,
        max_length=tokenizer_cfg["max_length"],
        stride=tokenizer_cfg["stride"],
    )

    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(split == "train"),
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
