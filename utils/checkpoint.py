import json
import shutil
from pathlib import Path

import torch


def save_checkpoint(model, tokenizer, optimizer, scheduler, step, epoch, val_loss, cfg):
    output_dir = Path(cfg["training"]["output_dir"])
    ckpt_dir = output_dir / f"checkpoint_step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
    model.config.to_json_file(str(ckpt_dir / "config.json"))
    tokenizer.save_pretrained(str(ckpt_dir / "tokenizer"))

    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

    with open(ckpt_dir / "trainer_state.json", "w") as f:
        json.dump({"step": step, "epoch": epoch, "val_loss": val_loss}, f)

    rotate_checkpoints(output_dir, cfg["training"]["save_total_limit"])
    return ckpt_dir


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    ckpt_dir = Path(checkpoint_path)

    state_dict = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)

    if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location="cpu")
        )

    if scheduler is not None and (ckpt_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(
            torch.load(ckpt_dir / "scheduler.pt", map_location="cpu")
        )

    with open(ckpt_dir / "trainer_state.json") as f:
        return json.load(f)


def rotate_checkpoints(output_dir, save_total_limit):
    output_dir = Path(output_dir)
    ckpts = sorted(
        output_dir.glob("checkpoint_step_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    while len(ckpts) > save_total_limit:
        shutil.rmtree(ckpts.pop(0))
