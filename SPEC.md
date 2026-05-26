# GPT-2 Post-Training Specification

> **Status:** Draft — edit before planning and implementation.  
> **Target model:** GPT-2 small (117 M / 124 M parameters, `gpt2` on Hugging Face)  
> **Goal:** Fine-tune GPT-2 small on a configurable text dataset, expose flexible sampling controls, and visualize both token embeddings and per-step generation probability distributions.

---

## Table of Contents

1. [Project Layout](#1-project-layout)
2. [Configuration System](#2-configuration-system)
3. [Data Pipeline](#3-data-pipeline)
4. [Tokenizer](#4-tokenizer)
5. [Model Setup](#5-model-setup)
6. [Post-Training (Fine-Tuning)](#6-post-training-fine-tuning)
7. [Sampling / Inference](#7-sampling--inference)
8. [Visualization](#8-visualization)
9. [Evaluation](#9-evaluation)
10. [Entry Points & CLI](#10-entry-points--cli)
11. [Dependencies](#11-dependencies)
12. [Open Questions](#12-open-questions)

---

## 1. Project Layout

```
GPT2-SDD/
├── SPEC.md                    # This file
├── config/
│   └── default.yaml           # All tuneable knobs with defaults
├── data/
│   ├── download.py            # Dataset download + cache
│   ├── preprocess.py          # Cleaning, splitting, encoding
│   └── dataset.py             # PyTorch Dataset / DataLoader factory
├── model/
│   ├── gpt2.py                # Model loading + weight init helpers
│   └── heads.py               # (Optional) task-specific heads
├── train.py                   # Training loop entry point
├── sample.py                  # Inference / generation entry point
├── visualize/
│   ├── embeddings.py          # Token-embedding visualization
│   └── distributions.py       # Per-step probability distribution plots
├── eval.py                    # Perplexity + optional BLEU/ROUGE
├── utils/
│   ├── logger.py              # Structured logging (wandb / tensorboard toggle)
│   └── checkpoint.py          # Save / load checkpoints
├── requirements.txt
└── README.md
```

> **Decision:** Flat layout (no `src/`). Dependency management via `pip` + `requirements.txt`.

---

## 2. Configuration System

All hyperparameters live in `config/default.yaml`. Runtime overrides via CLI flags using Hydra or argparse (choose one).

### 2.1 Dataset block

```yaml
data:
  # Primary toggle — change this URL to swap datasets entirely
  dataset_url: "https://huggingface.co/datasets/OEvortex/Bhagavad_Gita"
  dataset_name: "OEvortex/Bhagavad_Gita"   # HF datasets identifier
  subset: null                               # HF config/subset name if applicable
  split_column: null                         # Column that holds train/val/test label
  text_column: "English Translation"                        # Column containing raw text
  language_filter: "english"                 # Filter rows by language field (null = no filter)
  cache_dir: "./data/cache"
  train_ratio: 0.90
  val_ratio:   0.05
  test_ratio:  0.05
  seed: 42
```

### 2.2 Tokenizer block

```yaml
tokenizer:
  name: "gpt2"               # HF tokenizer identifier — must match model
  add_special_tokens: true
  max_length: 512            # Truncation ceiling per training example
  stride: 128                # Overlap for sliding-window chunking
  padding: "max_length"      # "max_length" | "longest" | false
  truncation: true
```

### 2.3 Model block

```yaml
model:
  name: "gpt2"               # HF model identifier (smallest, ~124 M params)
  from_pretrained: true      # true = load HF weights; false = random init
  freeze_layers: []          # List of layer indices to freeze, e.g. [0,1,2]
  dropout_override: null     # Override attention/residual dropout (null = use model defaults)
```

### 2.4 Training block

```yaml
training:
  output_dir: "./checkpoints"
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4   # effective batch = batch_size × grad_accum
  learning_rate: 5.0e-5
  lr_scheduler: "cosine"           # "linear" | "cosine" | "constant"
  warmup_ratio: 0.06               # fraction of total steps used for warmup
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: false                      # mixed precision
  bf16: false
  logging_steps: 50
  eval_steps: 200
  save_steps: 500
  save_total_limit: 3
  early_stopping_patience: 5      # stop if val loss doesn't improve for N evals
  seed: 42
  device: "auto"                  # "cpu" | "cuda" | "mps" | "auto"
```

### 2.5 Sampling block

```yaml
sampling:
  max_new_tokens: 200
  temperature: 1.0        # >1 = more random, <1 = more peaked
  top_k: 50               # 0 = disabled
  top_p: 0.95             # 1.0 = disabled (nucleus sampling)
  repetition_penalty: 1.0 # 1.0 = no penalty
  do_sample: true
  num_return_sequences: 1
  seed: null              # null = non-deterministic
```

### 2.6 Visualization block

```yaml
visualization:
  output_dir: "./viz"
  embeddings:
    enabled: true
    method: "umap"          # "umap" | "tsne" | "pca"
    n_components: 2         # 2 or 3
    n_neighbors: 15         # UMAP-specific
    min_dist: 0.1           # UMAP-specific
    perplexity: 30          # t-SNE-specific
    colormap: "tab20"
    max_tokens_to_plot: 500 # subsample vocabulary for legibility
    annotate_top_n: 50      # label the N most-frequent tokens
  distributions:
    enabled: true
    top_n_tokens: 20        # show top-N tokens in each bar chart
    save_gif: false         # stitch per-step plots into animated GIF
    dpi: 150
```

---

## 3. Data Pipeline

### 3.1 Download (`data/download.py`)

- Use `datasets.load_dataset(cfg.data.dataset_name, ...)` from the HuggingFace `datasets` library.
- The URL in `cfg.data.dataset_url` is informational/documentation; `dataset_name` is what `load_dataset` consumes.
- Cache to `cfg.data.cache_dir`; skip re-download if cache exists.
- Expose a `--force-download` flag that deletes cache and re-fetches.
- Log the dataset size (rows, columns, estimated bytes) after download.

### 3.2 Preprocessing (`data/preprocess.py`)

Steps applied in order:

1. **Column selection** — keep only `cfg.data.text_column`; drop all other columns.
2. **Language filter** — if `cfg.data.language_filter` is set, keep only rows where a `language` (or equivalent) column matches the filter value.
3. **Text cleaning**:
   - Strip leading/trailing whitespace.
   - Normalize Unicode to NFC.
   - Collapse multiple consecutive blank lines to one.
   - Remove control characters except `\n` and `\t`.
   - **Decision:** Keep verse numbering prefixes (e.g., `"BG 1.1:"`) as context — they provide structural signal to the model.
4. **Train / val / test split** — stratified random split using `cfg.data.*_ratio` and `cfg.data.seed`.
5. **Serialise** to Arrow/Parquet shards in `cfg.data.cache_dir/processed/`.

### 3.3 Dataset & DataLoader (`data/dataset.py`)

- Implement a `TextChunkDataset(torch.utils.data.Dataset)`:
  - Input: list of raw strings from a split.
  - Tokenize with sliding window: chunk size = `cfg.tokenizer.max_length`, stride = `cfg.tokenizer.stride`.
  - Each item: `{"input_ids": Tensor[L], "attention_mask": Tensor[L], "labels": Tensor[L]}` where `labels = input_ids` (causal LM).
  - Mask padding positions in `labels` with `-100` so they don't contribute to loss.
- `get_dataloader(split, cfg)` → `DataLoader` with `pin_memory=True` when CUDA is available.

---

## 4. Tokenizer

### 4.1 Loading

```python
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained(cfg.tokenizer.name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no native pad token
```

### 4.2 Byte-Pair Encoding (BPE) background

GPT-2's tokenizer uses BPE with a vocabulary of **50 257** tokens built from UTF-8 bytes. Characters not in the base vocabulary are represented as sequences of byte-level tokens, so it handles arbitrary Unicode without `[UNK]`.

### 4.3 Encoding contract

```
Input  : raw_string (str)
Output : {
    "input_ids":      LongTensor [seq_len],
    "attention_mask": LongTensor [seq_len],   # 1 = real token, 0 = padding
}
```

### 4.4 Special token handling

| Token | ID | Purpose |
|---|---|---|
| `<\|endoftext\|>` | 50256 | document boundary / pad surrogate |

No `[CLS]`, `[SEP]` tokens — GPT-2 is decoder-only.

---

## 5. Model Setup

### 5.1 Architecture (GPT-2 small)

| Hyper-param | Value |
|---|---|
| Layers (`n_layer`) | 12 |
| Attention heads (`n_head`) | 12 |
| Embedding dim (`n_embd`) | 768 |
| Context window (`n_ctx`) | 1024 |
| Vocabulary size | 50 257 |
| Total parameters | ~124 M |
| Activation | GELU |
| Positional encoding | Learned absolute |

### 5.2 Loading

```python
from transformers import GPT2LMHeadModel, GPT2Config

if cfg.model.from_pretrained:
    model = GPT2LMHeadModel.from_pretrained(cfg.model.name)
else:
    config = GPT2Config.from_pretrained(cfg.model.name)
    model = GPT2LMHeadModel(config)  # random weights
```

### 5.3 Layer freezing

If `cfg.model.freeze_layers` is non-empty, freeze `model.transformer.h[i]` for each listed index `i`. The embedding layer and LM head remain trainable unless explicitly added to the freeze list.

### 5.4 Dropout override

If `cfg.model.dropout_override` is set, iterate `model.config` and update `attn_pdrop`, `embd_pdrop`, `resid_pdrop` accordingly before instantiating.

---

## 6. Post-Training (Fine-Tuning)

### 6.1 Objective

Standard **causal language modelling (CLM)** — predict the next token given all preceding tokens. Loss = cross-entropy over non-padding positions:

```
L = -1/N Σ log P(token_t | token_{<t})
```

### 6.2 Training loop (`train.py`)

```
for epoch in range(cfg.training.epochs):
    for step, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss / cfg.training.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % cfg.training.eval_steps == 0:
            val_loss = evaluate(model, val_loader)
            log(val_loss)
            early_stopping_check(val_loss)

        if step % cfg.training.save_steps == 0:
            save_checkpoint(model, tokenizer, step)
```

### 6.3 Optimizer

`AdamW` from `torch.optim` or `transformers.AdamW`.  
Weight decay applied to all parameters **except** bias and LayerNorm weights.

### 6.4 LR Scheduler

`transformers.get_scheduler(cfg.training.lr_scheduler, ...)` with linear warmup for `warmup_ratio × total_steps` steps.

### 6.5 Mixed precision

Use `torch.cuda.amp.GradScaler` + `autocast` when `cfg.training.fp16 = true`.  
Use `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` when `cfg.training.bf16 = true`.

### 6.6 Checkpointing

- Save `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, step count, epoch, best val loss.
- Keep at most `cfg.training.save_total_limit` checkpoints; delete oldest on overflow.
- Save tokenizer alongside model for portability.

### 6.7 Logging

- Console: tqdm progress bar with live loss.
- File: JSON-lines log at `{output_dir}/train_log.jsonl`.
- Optional: Weights & Biases (`wandb`) or TensorBoard — toggled by a `logging.backend` config key.

---

## 7. Sampling / Inference

All sampling logic lives in `sample.py`. The model is loaded from a checkpoint or directly from `cfg.model.name`.

### 7.1 Greedy decoding (baseline)

```
next_token = argmax(logits[-1])
```

Used when `cfg.sampling.do_sample = false` and `top_k = 1`.

### 7.2 Temperature scaling

Applied **before** top-k / top-p filtering.

```python
logits = logits / cfg.sampling.temperature
probs  = F.softmax(logits, dim=-1)
```

- `temperature → 0`: distribution collapses to greedy.
- `temperature = 1`: unmodified model distribution.
- `temperature > 1`: flattened distribution (more diversity).

### 7.3 Top-k filtering

```python
if cfg.sampling.top_k > 0:
    k = min(cfg.sampling.top_k, logits.size(-1))
    threshold = logits.topk(k).values[..., -1, None]
    logits = logits.masked_fill(logits < threshold, float('-inf'))
```

Zero-out all tokens outside the top-k by mass before softmax.

### 7.4 Top-p (Nucleus) filtering

```python
if 0.0 < cfg.sampling.top_p < 1.0:
    sorted_logits, sorted_indices = logits.sort(descending=True)
    cumulative_probs = sorted_logits.softmax(-1).cumsum(-1)
    # remove tokens once cumulative prob exceeds top_p
    sorted_indices_to_remove = cumulative_probs - sorted_logits.softmax(-1) > cfg.sampling.top_p
    logits.scatter_(-1, sorted_indices, sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf')))
```

Keeps the smallest set of tokens whose cumulative probability ≥ `top_p`.

### 7.5 Repetition penalty

```python
if cfg.sampling.repetition_penalty != 1.0:
    for token_id in set(generated_ids):
        logits[token_id] /= cfg.sampling.repetition_penalty  # if logit > 0
        logits[token_id] *= cfg.sampling.repetition_penalty  # if logit < 0
```

### 7.6 Token sampling

```python
next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
```

### 7.7 Generation loop

```
generated = list(prompt_ids)
for _ in range(cfg.sampling.max_new_tokens):
    logits = model(torch.tensor([generated]))[:, -1, :]   # shape [vocab]
    logits = apply_temperature(logits)
    logits = apply_top_k(logits)
    logits = apply_top_p(logits)
    logits = apply_repetition_penalty(logits, generated)
    next_token = sample(logits)
    generated.append(next_token)
    if next_token == tokenizer.eos_token_id:
        break
    if cfg.visualization.distributions.enabled:
        record_distribution(logits, step=len(generated))
```

> **Decision:** Use `model.generate()` with custom `LogitsProcessor` subclasses. The `DistributionRecorder` processor captures post-filter logits at each step for visualization, giving the same hook points as a manual loop without re-implementing generation logic.

---

## 8. Visualization

### 8.1 Embedding Visualization (`visualize/embeddings.py`)

**What is visualized:** The static token embedding matrix `model.transformer.wte.weight` — shape `[vocab_size, n_embd]` = `[50257, 768]`. This is a snapshot of learned embedding geometry.

**Steps:**

1. Extract `wte_matrix = model.transformer.wte.weight.detach().cpu().numpy()`  — shape `[50257, 768]`.
2. Optionally restrict to the `cfg.visualization.embeddings.max_tokens_to_plot` most-frequent tokens in the training corpus (use token frequency table computed during tokenization).
3. Reduce to 2-D (or 3-D) with the selected method:
   - **PCA**: `sklearn.decomposition.PCA(n_components=2)`
   - **t-SNE**: `sklearn.manifold.TSNE(n_components=2, perplexity=cfg...perplexity)`
   - **UMAP**: `umap.UMAP(n_components=2, n_neighbors=cfg...n_neighbors, min_dist=cfg...min_dist)`
4. Color points by token category:
   - Punctuation
   - Digits / numerals
   - Common English words (cross-reference against a stopword list)
   - Subword pieces (tokens starting with `Ġ` in GPT-2 BPE notation)
   - Rare / other
5. Annotate top-`cfg.visualization.embeddings.annotate_top_n` tokens with their string representation.
6. Save figure to `{viz_dir}/embeddings_{method}.png` at `cfg.visualization.distributions.dpi` DPI.
7. Optionally render an interactive HTML plot via `plotly` for pan/zoom.

**Trigger points:**
- Before fine-tuning (pre-trained embeddings baseline).
- After fine-tuning (to observe domain shift, if any).
- **Decision:** Two snapshots only — once before fine-tuning (pretrained baseline) and once after. Not captured every epoch.

### 8.2 Per-Step Probability Distribution (`visualize/distributions.py`)

**What is visualized:** After the temperature / top-k / top-p filters are applied at each generation step, the probability distribution over the **top-N** tokens is plotted as a bar chart.

**Steps:**

1. At each decoding step, capture the filtered logit vector (post-temperature, post-top-k/top-p, before sampling).
2. Convert to probabilities via `softmax`.
3. Select top-`cfg.visualization.distributions.top_n_tokens` tokens by probability.
4. Render a horizontal bar chart:
   - X axis: probability (0–1).
   - Y axis: token string (decoded from ID).
   - Highlight the actually-sampled token in a different colour.
   - Title: `"Step {t}: generated '{token_str}'"`.
5. Save each step's chart to `{viz_dir}/dist_step_{t:04d}.png`.
6. If `cfg.visualization.distributions.save_gif = true`, use `PIL.Image` / `imageio` to stitch all per-step PNGs into `{viz_dir}/generation.gif`.

**Interactive mode (optional):** A `--interactive` CLI flag that opens a matplotlib window and pauses between steps so the user can inspect each distribution before advancing.

> **Decision:** Inference only. Capturing distributions during training adds significant storage overhead and is deferred; the inference-time visualization is sufficient to demonstrate the feature.

---

## 9. Evaluation

### 9.1 Perplexity

Computed on the test split after training:

```
PPL = exp(avg_cross_entropy_loss)
```

Lower is better. Report on both the full test set and per-verse (for the Bhagavad Gita dataset).

### 9.2 Qualitative samples

Generate N=10 completions from held-out verse prompts and log them alongside ground truth.

---

## 10. Entry Points & CLI

### `python train.py`

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Path to config file |
| `--override` | — | Hydra-style dot-notation overrides, e.g. `training.lr=1e-4` |
| `--resume` | `null` | Path to checkpoint to resume from |
| `--dry-run` | `false` | Run one batch, skip saving |

### `python sample.py`

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Config file |
| `--checkpoint` | `null` | Override checkpoint path |
| `--prompt` | `""` | Seed text |
| `--interactive` | `false` | Pause between steps for distribution inspection |

### `python visualize/embeddings.py`

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Config file |
| `--checkpoint` | `null` | Model weights to load (null = pretrained) |
| `--method` | from config | Override reduction method |

### `python eval.py`

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Config file |
| `--checkpoint` | required | Path to fine-tuned checkpoint |
| `--split` | `"test"` | Dataset split to evaluate on |

---

## 11. Dependencies

```
# requirements.txt (versions are starting points — pin after testing)

torch>=2.2.0
transformers>=4.40.0
datasets>=2.19.0
tokenizers>=0.19.0
accelerate>=0.29.0          # optional: multi-GPU / device placement
scikit-learn>=1.4.0         # PCA, t-SNE
umap-learn>=0.5.6           # UMAP
matplotlib>=3.8.0
plotly>=5.20.0              # interactive embedding plot
pillow>=10.3.0              # GIF stitching
imageio>=2.34.0             # GIF stitching
tqdm>=4.66.0
pyyaml>=6.0.1
numpy>=1.26.0
wandb>=0.16.0               # optional logging backend
```

> **Decision:** `argparse + PyYAML` only. No `hydra-core` dependency. CLI overrides use dot-notation parsed manually (e.g. `--override training.lr=1e-4`).

---

## 12. Open Questions

| # | Question | Status | Decision / Notes |
|---|---|---|---|
| 1 | Exact column name for English translation? | **Resolved** | Column name: `"English Translation"`|
| 2 | Pre-existing train/val/test splits? | **Resolved** | No — random 90/5/5 split created in preprocessing |
| 3 | Keep or strip verse metadata? | **Resolved** | Keep (e.g., `"BG 1.1:"` stays as context) |
| 4 | `model.generate()` vs manual loop? | **Resolved** | `model.generate()` + `LogitsProcessor` hooks |
| 5 | Embedding viz every N epochs? | **Resolved** | Before fine-tuning + after fine-tuning only |
| 6 | Distributions during training? | **Resolved** | Inference only |
| 7 | Hydra vs argparse + PyYAML? | **Resolved** | `argparse + PyYAML` |
| 8 | Multiple translations for BLEU/ROUGE? | **Resolved** | Assume single translation |
| 9 | Target hardware? | **Resolved** | Single GPU + CPU fallback (`device: "auto"`) |
| 10 | Interactive viz: block or thread? | **Resolved** | Block execution |
