"""Small native LoRA adapters for Z-Image attention projections."""

import math
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        # Train adapters in FP32, including when the frozen DiT uses BF16.
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        update = torch.nn.functional.linear(x.to(self.lora_a.dtype), self.lora_a)
        update = torch.nn.functional.linear(update, self.lora_b)
        result = self.base(x)
        return result + update.to(result.dtype) * self.scale


def install_transparency_lora(transformer, rank=32, alpha=32.0, targets=None):
    if rank < 1 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if any(isinstance(module, LoRALinear) for module in transformer.modules()):
        raise ValueError("LoRA is already installed on this transformer.")
    linear_modules = {name: module for name, module in transformer.named_modules() if isinstance(module, nn.Linear)}
    if targets is None:
        targets = [name for name in linear_modules
                   if name.endswith((".to_q", ".to_k", ".to_v", ".to_out.0"))]
    if not targets or len(set(targets)) != len(targets) or any(name not in linear_modules for name in targets):
        raise ValueError("LoRA targets must name existing, unique linear projections.")
    transformer.requires_grad_(False)
    for name in targets:
        parent_name, child_name = name.rsplit(".", 1)
        parent = transformer.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(linear_modules[name], rank, alpha))
    return {"rank": rank, "alpha": alpha, "targets": targets}


def save_transparency_lora(transformer, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lora_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in transformer.named_parameters()
             if name.endswith((".lora_a", ".lora_b"))}
    save_file(state, str(directory / "transformer_lora.safetensors"))


def load_transparency_lora(transformer, directory):
    directory = Path(directory)
    config = json.loads((directory / "lora_config.json").read_text(encoding="utf-8"))
    install_transparency_lora(transformer, **config)
    state = load_file(str(directory / "transformer_lora.safetensors"))
    expected = {name for name, _ in transformer.named_parameters() if name.endswith((".lora_a", ".lora_b"))}
    if set(state) != expected:
        raise ValueError("Incomplete or incompatible transparency LoRA checkpoint.")
    transformer.load_state_dict(state, strict=False)
    return config
