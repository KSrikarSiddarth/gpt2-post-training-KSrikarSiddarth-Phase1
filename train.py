import argparse
import contextlib
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import GPT2TokenizerFast, get_scheduler

from data.dataset import get_dataloader
from model.gpt2 import get_trainable_params, load_model, resolve_device
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import TrainingLogger


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, overrides):
    for kv in overrides:
        key, value = kv.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        last = keys[-1]
        if value == "null":
            d[last] = None
        elif value.lower() in ("true", "false"):
            d[last] = value.lower() == "true"
        else:
            try:
                d[last] = int(value)
            except ValueError:
                try:
                    d[last] = float(value)
                except ValueError:
                    d[last] = value
    return cfg


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, val_loader, device, autocast_ctx):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx:
            outputs = model(**batch)
        total_loss += outputs.loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def snapshot_embeddings(model, cfg, tag):
    try:
        from visualize.embeddings import snapshot
        snapshot(model, cfg, tag)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT_DIR")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_cfg(args.config), args.override)
    tcfg = cfg["training"]
    set_seeds(tcfg["seed"])

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["tokenizer"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_dataloader("train", cfg, tokenizer)
    val_loader = get_dataloader("val", cfg, tokenizer)

    model = load_model(cfg)
    device = resolve_device(cfg)

    param_groups = get_trainable_params(model)
    param_groups[0]["weight_decay"] = tcfg["weight_decay"]
    optimizer = torch.optim.AdamW(param_groups, lr=tcfg["learning_rate"])

    grad_accum = tcfg["gradient_accumulation_steps"]
    total_steps = (len(train_loader) // grad_accum) * tcfg["epochs"]
    warmup_steps = int(total_steps * tcfg["warmup_ratio"])
    scheduler = get_scheduler(
        tcfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_step, start_epoch, best_val_loss = 0, 0, float("inf")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_step = state["step"]
        start_epoch = state["epoch"]
        best_val_loss = state.get("val_loss", float("inf"))
        print(f"Resumed from step {start_step}, epoch {start_epoch}")

    use_fp16 = tcfg["fp16"] and torch.cuda.is_available()
    use_bf16 = tcfg["bf16"] and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    if use_fp16:
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    elif use_bf16:
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    logger = TrainingLogger(cfg)
    Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    snapshot_embeddings(model, cfg, "pretrained")

    global_step = start_step
    patience_counter = 0
    last_val_loss = best_val_loss
    should_stop = False

    model.train()
    for epoch in range(start_epoch, tcfg["epochs"]):
        if should_stop:
            break
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{tcfg['epochs']}") as pbar:
            for raw_step, batch in enumerate(pbar):
                if should_stop:
                    break

                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast_ctx:
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum

                scaler.scale(loss).backward()

                if (raw_step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), tcfg["max_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    train_loss = loss.item() * grad_accum
                    if global_step % tcfg["logging_steps"] == 0:
                        lr = scheduler.get_last_lr()[0]
                        logger.log({"train_loss": train_loss, "lr": lr}, step=global_step)
                        pbar.set_postfix(loss=f"{train_loss:.4f}", step=global_step)

                    if global_step % tcfg["eval_steps"] == 0:
                        last_val_loss = evaluate(model, val_loader, device, autocast_ctx)
                        logger.log({"val_loss": last_val_loss}, step=global_step)
                        print(f"\nStep {global_step}: val_loss={last_val_loss:.4f}")

                        if last_val_loss < best_val_loss:
                            best_val_loss = last_val_loss
                            patience_counter = 0
                            save_checkpoint(
                                model, tokenizer, optimizer, scheduler,
                                global_step, epoch, last_val_loss, cfg,
                            )
                        else:
                            patience_counter += 1
                            if patience_counter >= tcfg["early_stopping_patience"]:
                                print(f"Early stopping at step {global_step}")
                                should_stop = True
                                break

                    elif global_step % tcfg["save_steps"] == 0:
                        save_checkpoint(
                            model, tokenizer, optimizer, scheduler,
                            global_step, epoch, last_val_loss, cfg,
                        )

                if args.dry_run:
                    print("Dry run complete.")
                    logger.close()
                    return

    snapshot_embeddings(model, cfg, "finetuned")
    logger.close()
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to: {tcfg['output_dir']}")


if __name__ == "__main__":
    main()
