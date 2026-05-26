# Implementation Plan

> **Status:** Ready for execution  
> **Spec:** SPEC.md  
> **Do not start until this plan is approved.**

---

## Decisions Made

All open questions from SPEC.md §12 are resolved here. Items still unresolvable without
inspecting the dataset at runtime are marked **TODO**.

| # | Question | Decision |
|---|---|---|
| 1 | Exact column name for English translation |  `"English Translation"`|
| 2 | Pre-existing train/val/test splits? | **No** — create random 90/5/5 splits using `cfg.data.seed` |
| 3 | Keep or strip verse numbering prefixes? | **Keep** — e.g. `"BG 1.1:"` stays as context |
| 4 | `model.generate()` vs manual loop | **`model.generate()` with custom `LogitsProcessor` subclasses** |
| 5 | Embedding viz frequency | **Before fine-tuning + after fine-tuning only** (two snapshots) |
| 6 | Distributions during training? | **No** — inference only; training distributions omitted |
| 7 | Hydra vs argparse + PyYAML | **argparse + PyYAML** — no Hydra dependency |
| 8 | Multiple translations for BLEU/ROUGE? | **Assumed single translation** — skip BLEU/ROUGE; perplexity is primary metric |
| 9 | Target hardware | **Single GPU + CPU fallback** via `device: "auto"` |
| 10 | Interactive viz: block or thread? | **Block execution** — no threading complexity |
| — | Repo layout | **Flat layout**, pip + `requirements.txt` |
| — | Default logging backend | **Console + JSON-lines**; wandb off by default |

---

## Phase Overview

```
Phase 0  Scaffolding          directories, requirements.txt, config/default.yaml
Phase 1  Data Pipeline        download.py → preprocess.py → dataset.py
Phase 2  Model Setup          model/gpt2.py (load, freeze, dropout override)
Phase 3  Training             utils/logger.py, utils/checkpoint.py, train.py
Phase 4  Sampling             LogitsProcessor hooks, sample.py
Phase 5  Visualization        visualize/distributions.py, visualize/embeddings.py
Phase 6  Evaluation           eval.py
Phase 7  Docs                 README.md
```

Each phase depends on the previous. Within a phase, files can be written in any order unless
noted.

---

## Phase 0 — Scaffolding

### 0.1 Directory tree

Create empty `__init__.py` files so each package is importable:

```
config/
data/__init__.py
model/__init__.py
utils/__init__.py
visualize/__init__.py
checkpoints/   (gitignored)
viz/           (gitignored)
logs/          (gitignored)
data/cache/    (gitignored)
```

### 0.2 `requirements.txt`

Pin the versions listed in SPEC.md §11. Add `hydra-core` is **not** included.
Add `pytest>=7.4.0` and `pytest-cov` for testing.

```
torch>=2.2.0
transformers>=4.40.0
datasets>=2.19.0
tokenizers>=0.19.0
accelerate>=0.29.0
scikit-learn>=1.4.0
umap-learn>=0.5.6
matplotlib>=3.8.0
plotly>=5.20.0
pillow>=10.3.0
imageio>=2.34.0
tqdm>=4.66.0
pyyaml>=6.0.1
numpy>=1.26.0
wandb>=0.16.0
pytest>=7.4.0
pytest-cov>=5.0.0
```

### 0.3 `config/default.yaml`

Transcribe all six config blocks from SPEC.md §2 verbatim, with three changes:

1. `data.text_column` → `"translation"` (decision #1)
2. Add `logging.backend: "console"` block (`"console"` | `"wandb"` | `"tensorboard"`)
3. `data.split_column: null` remains (no pre-existing splits)

Full structure:

```yaml
data:
  dataset_url: "https://huggingface.co/datasets/OEvortex/Bhagavad_Gita"
  dataset_name: "OEvortex/Bhagavad_Gita"
  subset: null
  split_column: null
  text_column: "English Translation"
  language_filter: "english"
  cache_dir: "./data/cache"
  train_ratio: 0.90
  val_ratio: 0.05
  test_ratio: 0.05
  seed: 42

tokenizer:
  name: "gpt2"
  add_special_tokens: true
  max_length: 512
  stride: 128
  padding: "max_length"
  truncation: true

model:
  name: "gpt2"
  from_pretrained: true
  freeze_layers: []
  dropout_override: null

training:
  output_dir: "./checkpoints"
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  lr_scheduler: "cosine"
  warmup_ratio: 0.06
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: false
  bf16: false
  logging_steps: 50
  eval_steps: 200
  save_steps: 500
  save_total_limit: 3
  early_stopping_patience: 5
  seed: 42
  device: "auto"

sampling:
  max_new_tokens: 200
  temperature: 1.0
  top_k: 50
  top_p: 0.95
  repetition_penalty: 1.0
  do_sample: true
  num_return_sequences: 1
  seed: null

visualization:
  output_dir: "./viz"
  embeddings:
    enabled: true
    method: "umap"
    n_components: 2
    n_neighbors: 15
    min_dist: 0.1
    perplexity: 30
    colormap: "tab20"
    max_tokens_to_plot: 500
    annotate_top_n: 50
  distributions:
    enabled: true
    top_n_tokens: 20
    save_gif: false
    dpi: 150

logging:
  backend: "console"    # "console" | "wandb" | "tensorboard"
  log_dir: "./logs"
```

---

## Phase 1 — Data Pipeline

### 1.1 `data/download.py`

**Purpose:** Load the HF dataset, cache to disk, print dataset stats.

```
CLI args:
  --config   path to default.yaml
  --force    delete cache and re-download

Logic:
  1. Load cfg from yaml.
  2. If --force, delete cfg.data.cache_dir.
  3. Call datasets.load_dataset(cfg.data.dataset_name,
         cache_dir=cfg.data.cache_dir)
  4. Print: num rows, column names, estimated size in MB.
  5. Save raw dataset to cache_dir/raw/.
```

### 1.2 `data/preprocess.py`

**Purpose:** Clean text, split, serialise to Arrow shards.

```
Steps (in order):
  1. Load raw dataset from cache_dir/raw/.
  2. Column selection: keep only cfg.data.text_column.
     - Raise ValueError if column is absent (surface the TODO).
  3. Language filter: if cfg.data.language_filter is set,
     filter rows where `language` column == filter value.
     - If `language` column absent, skip filter with a warning.
  4. Text cleaning per row:
     a. unicodedata.normalize("NFC", text)
     b. text.strip()
     c. re.sub(r'\n{3,}', '\n\n', text)   # collapse blank lines
     d. remove control chars except \n and \t
     e. Keep verse prefixes (e.g., "BG 1.1:") — decision #3
  5. Random 90/5/5 split using cfg.data.seed.
     datasets.Dataset.train_test_split twice.
  6. Save splits to cache_dir/processed/{train,val,test}/ as Arrow.
```

### 1.3 `data/dataset.py`

**Purpose:** PyTorch Dataset wrapping tokenised sliding-window chunks.

```python
class TextChunkDataset(torch.utils.data.Dataset):
    """
    Tokenises a list of strings with sliding window.
    Each item: {input_ids, attention_mask, labels}
    labels == input_ids with padding positions set to -100.
    """

def get_dataloader(split: str, cfg) -> DataLoader:
    """
    Loads processed Arrow split, builds TextChunkDataset,
    returns DataLoader with pin_memory=True when CUDA available.
    """
```

Sliding-window chunking: chunk_size = `cfg.tokenizer.max_length`,
stride = `cfg.tokenizer.stride`. Each window is one dataset item.

---

## Phase 2 — Model Setup

### 2.1 `model/gpt2.py`

**Purpose:** Model loading helpers; freeze layers; dropout override.

```python
def load_model(cfg) -> GPT2LMHeadModel:
    """
    Loads GPT2LMHeadModel from pretrained or random init.
    Applies freeze_layers and dropout_override from cfg.
    Returns model on the resolved device.
    """

def resolve_device(cfg) -> torch.device:
    """
    cfg.training.device == "auto" → prefer CUDA > MPS > CPU.
    """

def get_trainable_params(model) -> list:
    """
    Returns params for AdamW: excludes bias and LayerNorm weights
    from weight_decay group.
    """
```

Layer freezing: iterate `model.transformer.h[i]` for each i in `cfg.model.freeze_layers`.
Dropout override: set `model.config.attn_pdrop`, `embd_pdrop`, `resid_pdrop` before init.

> **Note:** `model/heads.py` is deferred — no task-specific heads needed for CLM fine-tuning.

---

## Phase 3 — Training

### 3.1 `utils/logger.py`

**Purpose:** Unified logging facade.

```python
class TrainingLogger:
    def __init__(self, cfg): ...
    def log(self, metrics: dict, step: int): ...
    def close(self): ...
```

Backends:
- `"console"`: print JSON line to stdout + append to `{log_dir}/train_log.jsonl`.
- `"wandb"`: call `wandb.log(metrics, step=step)`.
- `"tensorboard"`: `SummaryWriter.add_scalar(...)`.

Only `"console"` backend needs to work for MVP.

### 3.2 `utils/checkpoint.py`

**Purpose:** Save, load, and rotate checkpoints.

```python
def save_checkpoint(model, tokenizer, optimizer, scheduler,
                    step, epoch, val_loss, cfg): ...

def load_checkpoint(checkpoint_path, model, optimizer=None,
                    scheduler=None) -> dict: ...

def rotate_checkpoints(output_dir, save_total_limit): ...
```

Checkpoint directory: `{output_dir}/checkpoint_step_{N}/`
Contents: `pytorch_model.bin`, `config.json`, `tokenizer/`, `trainer_state.json`

`rotate_checkpoints` deletes oldest directories when count exceeds `save_total_limit`.

### 3.3 `train.py`

**Purpose:** Training loop entry point.

```
CLI args:
  --config    config/default.yaml
  --override  key=value pairs (dot-notation, repeatable)
  --resume    path to checkpoint dir
  --dry-run   run one batch and exit

Logic:
  1. Parse args + load config + apply overrides.
  2. Set seeds (torch, numpy, random).
  3. Load tokenizer (pad_token = eos_token).
  4. Build DataLoaders for train and val.
  5. Load model via model/gpt2.py.
  6. Build AdamW optimizer with two param groups (wd / no-wd).
  7. Build LR scheduler via transformers.get_scheduler().
  8. If --resume, load_checkpoint().
  9. Viz: run embeddings.py snapshot BEFORE training.
  10. Training loop (pseudo-code from SPEC §6.2):
      - gradient accumulation
      - grad clipping
      - eval every eval_steps → early stopping check
      - checkpoint every save_steps
      - log every logging_steps
  11. After loop: run embeddings.py snapshot AFTER training.
  12. Print final val loss and checkpoint location.
```

Early stopping: maintain `best_val_loss` and `patience_counter`; raise `StopIteration` (or break) when counter exceeds `cfg.training.early_stopping_patience`.

Mixed precision: use `torch.cuda.amp.GradScaler` when `fp16=true`.

---

## Phase 4 — Sampling

### 4.1 `LogitsProcessor` subclasses (in `sample.py` or `utils/sampling.py`)

Implement as composable `transformers.LogitsProcessor` subclasses:

| Class | Applies |
|---|---|
| `TemperatureLogitsWarper` | divide by temperature |
| `TopKLogitsWarper` | zero out tokens outside top-k |
| `TopPLogitsWarper` | nucleus filtering |
| `RepetitionPenaltyLogitsProcessor` | penalise already-generated tokens |
| `DistributionRecorder` | side-effect: captures post-filter logits per step |

> `TemperatureLogitsWarper`, `TopKLogitsWarper`, `TopPLogitsWarper` are already implemented in
> `transformers`. Reuse them; only implement `RepetitionPenaltyLogitsProcessor` and
> `DistributionRecorder` from scratch.

### 4.2 `sample.py`

**Purpose:** Inference entry point.

```
CLI args:
  --config       config/default.yaml
  --checkpoint   path to fine-tuned checkpoint (null = pretrained weights)
  --prompt       seed text string
  --interactive  pause between steps for distribution inspection

Logic:
  1. Load cfg + model + tokenizer.
  2. Encode prompt.
  3. Build LogitsProcessorList from cfg.sampling.
  4. If cfg.visualization.distributions.enabled:
       attach DistributionRecorder to processor list.
  5. Call model.generate(..., logits_processor=processor_list).
  6. Decode and print output.
  7. If distributions enabled: call visualize/distributions.py to render plots.
  8. If --interactive: pause after each step (DistributionRecorder triggers plt.show()).
```

---

## Phase 5 — Visualization

### 5.1 `visualize/distributions.py`

**Purpose:** Render per-step probability bar charts; optionally stitch GIF.

```python
def plot_step_distribution(logits, step, sampled_token_id,
                            tokenizer, cfg): ...

def save_gif(viz_dir): ...
```

Each chart:
- Horizontal bar chart, top-N tokens by probability.
- Sampled token highlighted in orange; others in steelblue.
- Saved as `{viz_dir}/dist_step_{t:04d}.png`.

GIF stitching uses `imageio.mimsave` if `cfg.visualization.distributions.save_gif = true`.

### 5.2 `visualize/embeddings.py`

**Purpose:** Extract `wte` matrix, reduce to 2D, annotate, save PNG + optional HTML.

```
CLI args:
  --config      config/default.yaml
  --checkpoint  model weights (null = pretrained)
  --method      override reduction method

Steps:
  1. Load model, extract wte_matrix [50257, 768].
  2. Subsample to max_tokens_to_plot most-frequent tokens
     (frequency table from training corpus; if not available, use random sample).
  3. Reduce with selected method (PCA / t-SNE / UMAP).
  4. Categorise each token: Punctuation | Digit | Stopword | Subword (Ġ prefix) | Other.
  5. Scatter plot coloured by category; annotate top-N by frequency.
  6. Save PNG to {viz_dir}/embeddings_{method}_{tag}.png
     where tag is "pretrained" or "finetuned".
  7. Optionally save interactive plotly HTML.
```

Token frequency table: computed during Phase 1 preprocessing and saved to
`data/cache/token_freq.json` for reuse here.

---

## Phase 6 — Evaluation

### 6.1 `eval.py`

**Purpose:** Compute perplexity on test split; optional qualitative samples.

```
CLI args:
  --config      config/default.yaml
  --checkpoint  required — path to fine-tuned checkpoint
  --split       "test" (default)

Logic:
  1. Load model + tokenizer from checkpoint.
  2. Build DataLoader for requested split.
  3. Compute avg cross-entropy loss over all batches.
  4. PPL = exp(avg_loss). Print and save to {log_dir}/eval_results.json.
  5. Per-verse PPL: group test rows by source verse, compute individual PPLs.
  6. Qualitative: generate 10 completions from held-out verse prompts.
     Log prompt + generation + ground-truth to {log_dir}/qualitative_samples.json.
```

BLEU/ROUGE/BERTScore: **skipped** (decision #8, single translation per verse).
TODO: If multiple translations are discovered at runtime, add optional `--bleu` flag.

---

## Phase 7 — Documentation

### 7.1 `README.md`

Cover: project overview, quickstart, config reference, training tips, example outputs.
Write last — after all phases are working.

---

## File Dependency Graph

```
config/default.yaml
    └── consumed by all entry points

data/download.py
    └── produces data/cache/raw/

data/preprocess.py
    ├── depends on: data/download.py output
    └── produces data/cache/processed/, data/cache/token_freq.json

data/dataset.py
    └── depends on: data/cache/processed/

model/gpt2.py
    └── depends on: config

utils/logger.py          (no internal deps)
utils/checkpoint.py      (no internal deps)

train.py
    ├── depends on: data/dataset.py, model/gpt2.py,
    │               utils/logger.py, utils/checkpoint.py
    └── calls: visualize/embeddings.py (before + after)

visualize/distributions.py
    └── no internal deps beyond config + tokenizer

sample.py
    ├── depends on: model/gpt2.py, utils/checkpoint.py
    └── calls: visualize/distributions.py

visualize/embeddings.py
    └── depends on: model/gpt2.py, data/cache/token_freq.json

eval.py
    └── depends on: data/dataset.py, model/gpt2.py, utils/checkpoint.py
```

---

## TODOs Before First Run

1. **`data.text_column`**: Load the dataset interactively and run
   `print(ds.features)` to confirm the English column name.
   Currently assumed `"translation"`.

2. **Language filter column**: Check whether a `language` column exists.
   If not, `language_filter` config will be silently skipped.

3. **BERTScore**: Decide if semantic similarity evaluation is in scope.
   If yes, add `bert-score>=0.3.13` to requirements and implement in `eval.py`.

4. **Dataset auth**: Check if `OEvortex/Bhagavad_Gita` requires a HuggingFace
   login token. If so, add `huggingface_hub login` step to docs.

5. **`data/cache/token_freq.json`**: Frequency table is needed by `visualize/embeddings.py`.
   If preprocessing is skipped, embeddings.py must fall back to random subsampling.

---

## Out of Scope (for this plan)

- Multi-GPU / distributed training (`accelerate` is listed as a dependency but not wired up).
- Task-specific heads (`model/heads.py` deferred).
- BLEU / ROUGE / BERTScore evaluation (single-translation dataset assumed).
- TensorBoard logging backend (only console + wandb stubs).
- 3-D embedding plots (`n_components: 3` in config is parsed but only 2-D rendering implemented).
