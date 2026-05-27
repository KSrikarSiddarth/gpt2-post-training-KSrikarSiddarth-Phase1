import argparse
import json
import random
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now",
})

_CATS = ["Subword", "Punctuation", "Digit", "Stopword", "Other"]
_COLORS = dict(zip(_CATS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_freq(cache_dir):
    p = Path(cache_dir) / "token_freq.json"
    if p.exists():
        with open(p) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def _subsample(freq, max_tokens, vocab_size):
    if freq:
        return sorted(freq, key=freq.get, reverse=True)[:max_tokens]
    return random.sample(range(vocab_size), min(max_tokens, vocab_size))


def _categorize(raw_tok):
    """Classify a raw BPE token string (with Ġ prefix intact)."""
    if raw_tok.startswith("Ġ"):   # Ġ = GPT-2 word-boundary marker
        return "Subword"
    s = raw_tok.strip()
    if not s:
        return "Other"
    if all(c in string.punctuation for c in s):
        return "Punctuation"
    if s.isdigit():
        return "Digit"
    if s.lower() in STOPWORDS:
        return "Stopword"
    return "Other"


def _reduce(matrix, method, cfg):
    ecfg = cfg["visualization"]["embeddings"]
    nc = ecfg["n_components"]
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=nc, random_state=0).fit_transform(matrix)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=nc, perplexity=ecfg["perplexity"], random_state=0
        ).fit_transform(matrix)
    if method == "umap":
        import umap as umap_lib
        return umap_lib.UMAP(
            n_components=nc,
            n_neighbors=ecfg["n_neighbors"],
            min_dist=ecfg["min_dist"],
            random_state=0,
        ).fit_transform(matrix)
    raise ValueError(f"Unknown reduction method: '{method}'")


def _plot(coords, token_ids, disp_toks, categories, freq, cfg, tag, method):
    ecfg = cfg["visualization"]["embeddings"]
    viz_dir = Path(cfg["visualization"]["output_dir"])
    viz_dir.mkdir(parents=True, exist_ok=True)

    cats_arr = np.array(categories)
    fig, ax = plt.subplots(figsize=(12, 10))
    for cat in _CATS:
        mask = cats_arr == cat
        if mask.any():
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                label=cat, color=_COLORS[cat], alpha=0.6, s=8,
            )

    if freq:
        top_set = set(sorted(freq, key=freq.get, reverse=True)[: ecfg["annotate_top_n"]])
        for i, tid in enumerate(token_ids):
            if tid in top_set:
                ax.annotate(
                    repr(disp_toks[i]),
                    (coords[i, 0], coords[i, 1]),
                    fontsize=6, alpha=0.8,
                )

    ax.legend(markerscale=2, fontsize=9)
    ax.set_title(f"GPT-2 token embeddings — {method.upper()} — {tag}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()

    out_png = viz_dir / f"embeddings_{method}_{tag}.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_png}")

    try:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame({
            "x": coords[:, 0], "y": coords[:, 1],
            "token": disp_toks, "category": categories,
        })
        html_fig = px.scatter(
            df, x="x", y="y", color="category", hover_data=["token"],
            title=f"GPT-2 embeddings — {method.upper()} — {tag}",
        )
        out_html = viz_dir / f"embeddings_{method}_{tag}.html"
        html_fig.write_html(str(out_html))
        print(f"Saved → {out_html}")
    except ImportError:
        pass


def _run(model, cfg, tag, method=None):
    from transformers import GPT2TokenizerFast

    ecfg = cfg["visualization"]["embeddings"]
    method = method or ecfg["method"]

    wte = model.transformer.wte.weight.detach().cpu().numpy()
    vocab_size = wte.shape[0]

    freq = _load_freq(cfg["data"]["cache_dir"])
    token_ids = _subsample(freq, ecfg["max_tokens_to_plot"], vocab_size)
    sub_matrix = wte[token_ids]

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["tokenizer"]["name"])
    raw_toks = [tokenizer.convert_ids_to_tokens(tid) for tid in token_ids]
    disp_toks = [tokenizer.decode([tid]) for tid in token_ids]
    categories = [_categorize(r) for r in raw_toks]

    coords = _reduce(sub_matrix, method, cfg)
    _plot(coords, token_ids, disp_toks, categories, freq, cfg, tag, method)


def snapshot(model, cfg, tag):
    """Public API for train.py — called before and after fine-tuning."""
    _run(model, cfg, tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None, metavar="CHECKPOINT_DIR")
    parser.add_argument("--method", default=None, choices=["pca", "tsne", "umap"])
    parser.add_argument("--tag", default="manual")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    from model.gpt2 import load_model
    model = load_model(cfg)

    if args.checkpoint:
        from utils.checkpoint import load_checkpoint
        load_checkpoint(args.checkpoint, model)

    method = args.method or cfg["visualization"]["embeddings"]["method"]
    _run(model, cfg, args.tag, method=method)


if __name__ == "__main__":
    main()
