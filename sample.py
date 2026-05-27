import argparse
from pathlib import Path

import torch
import yaml
from transformers import (
    GPT2TokenizerFast,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from model.gpt2 import load_model, resolve_device
from utils.checkpoint import load_checkpoint
from utils.sampling import DistributionRecorder, RepetitionPenaltyLogitsProcessor


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_processor_list(cfg, recorder=None):
    scfg = cfg["sampling"]
    processors = []
    if scfg["repetition_penalty"] != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(scfg["repetition_penalty"]))
    if scfg["temperature"] != 1.0:
        processors.append(TemperatureLogitsWarper(scfg["temperature"]))
    if scfg["top_k"] > 0:
        processors.append(TopKLogitsWarper(scfg["top_k"]))
    if 0.0 < scfg["top_p"] < 1.0:
        processors.append(TopPLogitsWarper(scfg["top_p"]))
    if recorder is not None:
        processors.append(recorder)
    return LogitsProcessorList(processors)


def render_distributions(recorder, sampled_ids, tokenizer, cfg):
    try:
        from visualize.distributions import plot_step_distribution
        viz_dir = Path(cfg["visualization"]["output_dir"])
        viz_dir.mkdir(parents=True, exist_ok=True)
        for step, scores in enumerate(recorder.records):
            token_id = sampled_ids[step] if step < len(sampled_ids) else None
            plot_step_distribution(scores, step, token_id, tokenizer, cfg)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None, metavar="CHECKPOINT_DIR")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    scfg = cfg["sampling"]
    device = resolve_device(cfg)

    if args.checkpoint:
        tokenizer = GPT2TokenizerFast.from_pretrained(
            str(Path(args.checkpoint) / "tokenizer")
        )
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(cfg["tokenizer"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(cfg)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model)
    model.eval()

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    recorder = None
    if cfg["visualization"]["distributions"]["enabled"]:
        def on_step(step, scores):
            if args.interactive:
                try:
                    import matplotlib.pyplot as plt
                    from visualize.distributions import plot_step_distribution
                    plot_step_distribution(scores, step, None, tokenizer, cfg)
                    plt.show()
                    input(f"Step {step} — press Enter to continue...")
                except ImportError:
                    pass
        recorder = DistributionRecorder(on_step=on_step)

    processor_list = build_processor_list(cfg, recorder)

    if scfg.get("seed") is not None:
        torch.manual_seed(scfg["seed"])

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=scfg["max_new_tokens"],
            do_sample=scfg["do_sample"],
            num_return_sequences=scfg["num_return_sequences"],
            logits_processor=processor_list,
            pad_token_id=tokenizer.eos_token_id,
            # Neutral values so generate() adds no built-in processors
            # that would double-apply our custom ones above.
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )

    prompt_len = input_ids.shape[-1]
    for i, seq in enumerate(output_ids):
        print(f"\n--- Generation {i + 1} ---")
        print(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))

    if recorder is not None and recorder.records and not args.interactive:
        sampled_ids = output_ids[0][prompt_len:].tolist()
        render_distributions(recorder, sampled_ids, tokenizer, cfg)


if __name__ == "__main__":
    main()
