from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_step_distribution(scores, step, sampled_token_id, tokenizer, cfg):
    dist_cfg = cfg["visualization"]["distributions"]
    viz_dir = Path(cfg["visualization"]["output_dir"])
    viz_dir.mkdir(parents=True, exist_ok=True)

    if scores.dim() == 2:
        scores = scores[0]

    probs = torch.softmax(scores.float(), dim=-1)
    top_n = dist_cfg["top_n_tokens"]
    top_probs, top_ids = probs.topk(top_n)
    top_probs = top_probs.numpy()
    top_ids = top_ids.tolist()

    token_strs = [repr(tokenizer.decode([tid])) for tid in top_ids]
    colors = [
        "orange" if tid == sampled_token_id else "steelblue"
        for tid in top_ids
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(top_n), top_probs[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(token_strs[::-1], fontsize=8)
    ax.set_xlabel("Probability")
    ax.set_xlim(0, max(top_probs) * 1.15)

    sampled_str = (
        repr(tokenizer.decode([sampled_token_id]))
        if sampled_token_id is not None
        else "?"
    )
    ax.set_title(f"Step {step}: generated {sampled_str}")
    plt.tight_layout()

    out = viz_dir / f"dist_step_{step:04d}.png"
    plt.savefig(out, dpi=dist_cfg["dpi"])
    plt.close(fig)


def save_gif(viz_dir):
    import imageio

    viz_dir = Path(viz_dir)
    frames = sorted(viz_dir.glob("dist_step_*.png"))
    if not frames:
        print("No dist_step_*.png files found — nothing to stitch.")
        return
    images = [imageio.imread(str(f)) for f in frames]
    out = viz_dir / "generation.gif"
    imageio.mimsave(str(out), images, fps=2)
    print(f"Saved GIF → {out}")
