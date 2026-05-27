import json
from pathlib import Path


class TrainingLogger:
    def __init__(self, cfg):
        log_cfg = cfg["logging"]
        self.backend = log_cfg["backend"]
        self.log_dir = Path(log_cfg["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.log_dir / "train_log.jsonl"

        if self.backend == "wandb":
            import wandb
            wandb.init(project="gpt2-bhagavad-gita", config=cfg)
            self._wandb = wandb
        elif self.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log(self, metrics: dict, step: int):
        if self.backend == "console":
            record = {"step": step, **metrics}
            print(json.dumps(record))
            with open(self._jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")
        elif self.backend == "wandb":
            self._wandb.log(metrics, step=step)
        elif self.backend == "tensorboard":
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, global_step=step)

    def close(self):
        if self.backend == "wandb":
            self._wandb.finish()
        elif self.backend == "tensorboard":
            self._writer.close()
