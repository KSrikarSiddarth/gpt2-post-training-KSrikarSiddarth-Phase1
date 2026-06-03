import torch
from transformers import GPT2Config, GPT2LMHeadModel


def resolve_device(cfg):
    setting = cfg["training"]["device"]
    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(setting)


def load_model(cfg):
    model_cfg = cfg["model"]
    dropout = model_cfg.get("dropout_override")

    gpt2_config = GPT2Config.from_pretrained(model_cfg["name"])
    if dropout is not None:
        gpt2_config.attn_pdrop = dropout
        gpt2_config.embd_pdrop = dropout
        gpt2_config.resid_pdrop = dropout

    if model_cfg["from_pretrained"]:
        model = GPT2LMHeadModel.from_pretrained(model_cfg["name"], config=gpt2_config)
    else:
        model = GPT2LMHeadModel(gpt2_config)

    for i in model_cfg.get("freeze_layers") or []:
        for param in model.transformer.h[i].parameters():
            param.requires_grad = False

    # transformers >= 4.46 warns when loss_type is None (absent in old GPT-2 configs)
    if getattr(model.config, "loss_type", None) is None:
        model.config.loss_type = "ForCausalLMLoss"

    return model.to(resolve_device(cfg))


def get_trainable_params(model):
    no_decay = {"bias", "LayerNorm.weight"}
    decay, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
