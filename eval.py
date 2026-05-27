import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from data.dataset import get_dataloader
from model.gpt2 import load_model, resolve_device
from utils.checkpoint import load_checkpoint


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def compute_perplexity(model, dataloader, device):
    """Token-weighted perplexity over all non-padding positions in the split."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in tqdm(dataloader, desc="Perplexity"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        n_tokens = (batch["labels"] != -100).sum().item()
        total_nll += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    return math.exp(total_nll / max(total_tokens, 1))


@torch.no_grad()
def compute_verse_ppl(model, tokenizer, text, device, max_length):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    outputs = model(input_ids=input_ids, labels=input_ids.clone())
    return math.exp(outputs.loss.item())


def make_prompt(text, tokenizer, n_tokens=30):
    ids = tokenizer.encode(text)
    n = min(n_tokens, max(1, len(ids) // 2))
    return tokenizer.decode(ids[:n])


@torch.no_grad()
def generate_completion(model, tokenizer, prompt, device, cfg):
    scfg = cfg["sampling"]
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=scfg["do_sample"],
        temperature=scfg["temperature"],
        top_k=scfg["top_k"],
        top_p=scfg["top_p"],
        repetition_penalty=scfg["repetition_penalty"],
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", required=True, metavar="CHECKPOINT_DIR")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = resolve_device(cfg)
    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    tok_path = Path(args.checkpoint) / "tokenizer"
    tokenizer = GPT2TokenizerFast.from_pretrained(
        str(tok_path) if tok_path.exists() else cfg["tokenizer"]["name"]
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(cfg)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    # 1. Full-split perplexity (chunk-level, token-weighted)
    dataloader = get_dataloader(args.split, cfg, tokenizer)
    ppl = compute_perplexity(model, dataloader, device)
    print(f"\n{args.split} perplexity: {ppl:.4f}")

    # 2. Per-verse perplexity
    cache_dir = Path(cfg["data"]["cache_dir"])
    text_col = cfg["data"]["text_column"]
    ds = load_from_disk(str(cache_dir / "processed" / args.split))
    verses = ds[text_col]

    per_verse = []
    for i, text in enumerate(tqdm(verses, desc="Per-verse PPL")):
        try:
            verse_ppl = compute_verse_ppl(
                model, tokenizer, text, device, cfg["tokenizer"]["max_length"]
            )
        except Exception:
            verse_ppl = float("inf")
        per_verse.append({"verse_idx": i, "text": text[:80], "ppl": verse_ppl})

    finite = [r["ppl"] for r in per_verse if math.isfinite(r["ppl"])]
    avg_verse_ppl = sum(finite) / max(len(finite), 1)
    print(f"Per-verse PPL (mean): {avg_verse_ppl:.4f}")

    eval_results = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "perplexity": ppl,
        "per_verse_mean_ppl": avg_verse_ppl,
        "num_verses": len(verses),
        "per_verse": per_verse,
    }
    eval_path = log_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved → {eval_path}")

    # 3. Qualitative samples (N=10, spread across the split)
    n_samples = min(10, len(verses))
    step = max(1, len(verses) // n_samples)
    indices = list(range(0, len(verses), step))[:n_samples]

    qual_samples = []
    for idx in tqdm(indices, desc="Qualitative samples"):
        text = verses[idx]
        prompt = make_prompt(text, tokenizer, n_tokens=30)
        generation = generate_completion(model, tokenizer, prompt, device, cfg)
        qual_samples.append(
            {"verse_idx": idx, "prompt": prompt, "generation": generation, "ground_truth": text}
        )

    qual_path = log_dir / "qualitative_samples.json"
    with open(qual_path, "w") as f:
        json.dump(qual_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved → {qual_path}")

    print("\n--- Sample outputs ---")
    for s in qual_samples[:3]:
        print(f"\nPrompt:     {s['prompt']}")
        print(f"Generation: {s['generation'][:120]}")
        print(f"Truth:      {s['ground_truth'][:120]}")


if __name__ == "__main__":
    main()
