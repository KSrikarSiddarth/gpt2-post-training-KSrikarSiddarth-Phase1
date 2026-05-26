# GPT-2 Post-Training — CLAUDE.md

## What This Project Does

Fine-tunes GPT-2 small (124 M parameters) on the Bhagavad Gita English translation dataset.
Exposes configurable sampling controls (temperature, top-k, top-p, repetition penalty) and
produces two kinds of visualization: token-embedding geometry plots and per-step generation
probability distribution bar charts.

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Deep learning | PyTorch 2.2+ | Native AMP, standard ecosystem |
| Model / tokenizer | HuggingFace `transformers` | GPT-2 weights + tokenizer in one call |
| Dataset loading | HuggingFace `datasets` | Arrow-backed caching, streaming support |
| Config | argparse + PyYAML | No heavy dependency (Hydra rejected) |
| Dim-reduction | scikit-learn (PCA/t-SNE), umap-learn | Standard visualization stack |
| Viz | matplotlib + plotly | Static PNGs + optional interactive HTML |
| Logging | JSON-lines + tqdm; wandb optional | wandb off by default, toggled via config |

## Key Design Decisions

- **Config system**: `config/default.yaml` is the single source of truth. CLI overrides use
  dot-notation argparse flags (e.g. `--override training.lr=1e-4`).
- **Generation**: `model.generate()` with custom `LogitsProcessor` subclasses for temperature,
  top-k, top-p, and repetition penalty. Cleaner than a manual loop and lets visualization hooks
  inject at each step via `LogitsProcessor`.
- **Embedding visualization**: two snapshots only — before fine-tuning (pretrained baseline) and
  after. Not captured every epoch.
- **Distribution visualization**: inference only. Capturing distributions during training is
  omitted to avoid large storage overhead.
- **Verse metadata**: kept in text (e.g., `"BG 1.1:"`). Gives the model structural context at
  no extra preprocessing cost.
- **Dataset splits**: random 90 / 5 / 5 train/val/test. The HF dataset has no pre-existing splits.
- **Target hardware**: single GPU with automatic CPU fallback (`device: "auto"`). `fp16` disabled
  by default; enable manually for CUDA training.

## Directory Layout

```
.
├── CLAUDE.md
├── PLAN.md
├── SPEC.md
├── config/
│   └── default.yaml
├── data/
│   ├── download.py        # HF datasets.load_dataset + cache
│   ├── preprocess.py      # clean, split, serialise to Arrow
│   └── dataset.py         # TextChunkDataset + DataLoader factory
├── model/
│   └── gpt2.py            # load, freeze layers, dropout override
├── utils/
│   ├── logger.py          # JSON-lines + optional wandb
│   └── checkpoint.py      # save / load / rotate checkpoints
├── visualize/
│   ├── embeddings.py      # wte matrix → PCA/t-SNE/UMAP plot
│   └── distributions.py   # per-step probability bar charts + GIF
├── train.py
├── sample.py
├── eval.py
└── requirements.txt
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download + preprocess the dataset
python data/download.py
python data/preprocess.py

# 3. Train (resumes from checkpoint if --resume is supplied)
python train.py --config config/default.yaml

# 4. Generate text from a prompt
python sample.py --checkpoint checkpoints/best --prompt "Arjuna said:"

# 5. Visualize token embeddings (pre- or post-training)
python visualize/embeddings.py --checkpoint checkpoints/best

# 6. Evaluate perplexity on the test split
python eval.py --checkpoint checkpoints/best
```

## Conventions

- All tunable knobs live in `config/default.yaml`. Never hardcode hyperparameters in source files.
- Output directories: `./checkpoints/` (model weights), `./viz/` (plots), `./logs/` (JSON-lines).
- `labels` tensors use `-100` for padding positions (PyTorch cross-entropy ignore index).
- Checkpoint filenames: `checkpoint_step_{N}/` with `pytorch_model.bin`, `tokenizer/`, `trainer_state.json`.
- One `LogitsProcessor` subclass per sampling modifier — keep them composable and independently testable.

## TODOs Before First Run

- Verify exact column name for English translation in `OEvortex/Bhagavad_Gita` (assumed `"translation"`).
- Confirm whether BERTScore evaluation is in scope.
- Check if the dataset exposes a `language` column (needed for `language_filter`).
