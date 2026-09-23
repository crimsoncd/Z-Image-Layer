"""PEFT low-level LoRA injection for the native Z-Image transformer.

Reference: https://huggingface.co/docs/peft/main/en/developer_guides/low_level_api
Checkpoints are adapters, not complete models, and are separate from the older
transparency_lora.py checkpoint format.
"""

import json
from pathlib import Path
import re

import peft
from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
import torch
from torch import nn


GROUPS = ("attention", "mlp", "modulation", "input", "output", "time", "text")
SCOPES = ("layers", "noise_refiner", "context_refiner")


def csv_values(value, choices):
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or set(result) - set(choices):
        raise ValueError(f"Expected comma-separated choices from {choices}; got {value!r}")
    return set(result)


def block_indices(value, count):
    if value is None:
        return set(range(count))
    result = set()
    for item in value.split(","):
        bounds = item.strip().split("-")
        if len(bounds) == 1:
            result.add(int(bounds[0]))
        elif len(bounds) == 2:
            start, end = map(int, bounds)
            if end < start:
                raise ValueError("Layer ranges must be increasing.")
            result.update(range(start, end + 1))
        else:
            raise ValueError("--blocks expects indices/ranges, e.g. 0-3,20,25-29")
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"Main block indices must be within 0..{count - 1}.")
    return result


def resolve_targets(model, groups="attention,mlp", scopes="layers,noise_refiner",
                    blocks=None, target_regex=None, exclude_regex=None):
    """Return explicit Linear names. Regex mode replaces groups/scopes/blocks.

    Groups input/output target only the active 2D patch (2,1) projections used
    by the existing pipeline, not inactive alternative patch-size heads.
    """
    selected_groups = csv_values(groups, GROUPS)
    selected_scopes = csv_values(scopes, SCOPES)
    indices = block_indices(blocks, len(model.layers))
    custom = re.compile(target_regex) if target_regex else None
    excluded = re.compile(exclude_regex) if exclude_regex else None
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if custom:
            take = custom.fullmatch(name) is not None
        else:
            parts = name.split(".")
            in_scope = parts[0] in selected_scopes
            if parts[0] == "layers":
                in_scope = in_scope and int(parts[1]) in indices
            take = in_scope and (
                ("attention" in selected_groups and name.endswith((".to_q", ".to_k", ".to_v", ".to_out.0")))
                or ("mlp" in selected_groups and ".feed_forward." in name)
                or ("modulation" in selected_groups and ".adaLN_modulation." in name)
            )
            take = take or ("input" in selected_groups and name == "all_x_embedder.2-1")
            take = take or ("output" in selected_groups and name == "all_final_layer.2-1.linear")
            take = take or ("time" in selected_groups and name.startswith("t_embedder."))
            take = take or ("text" in selected_groups and name.startswith("cap_embedder."))
        if take and not (excluded and excluded.search(name)):
            targets.append(name)
    if not targets:
        raise ValueError("No Linear modules matched the LoRA selection.")
    return sorted(targets)


def inject_lora(model, config):
    if getattr(model, "peft_config", None):
        raise ValueError("Expected a fresh transformer without existing PEFT adapters.")
    model.requires_grad_(False)
    model = inject_adapter_in_model(config, model, adapter_name="default")
    # Keep optimizer parameters FP32 even when the frozen transformer is BF16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def create_lora(model, targets, rank=32, alpha=32.0, dropout=0.0, use_dora=False, use_rslora=False):
    if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("Invalid LoRA rank, alpha or dropout.")
    # A regex gives exact matching; PEFT list matching also accepts suffixes.
    pattern = "(?:" + "|".join(re.escape(name) for name in targets) + ")"
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules=pattern, bias="none", init_lora_weights=True,
                        use_dora=use_dora, use_rslora=use_rslora)
    return inject_lora(model, config)


def save_adapter(model, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.peft_config["default"].save_pretrained(directory)
    state = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    save_file({name: value.detach().cpu().contiguous() for name, value in state.items()},
              str(directory / "adapter_model.safetensors"))
    metadata = {**metadata, "format": "zimage-rgba-peft-v1", "peft_version": peft.__version__}
    (directory / "rgba_peft.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_adapter(model, directory, trainable=False):
    directory = Path(directory)
    metadata = json.loads((directory / "rgba_peft.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "zimage-rgba-peft-v1":
        raise ValueError("Not a native RGBA PEFT checkpoint. Legacy LoRA checkpoints are not interchangeable.")
    config = LoraConfig.from_pretrained(directory)
    config.inference_mode = False
    model = inject_lora(model, config)
    state = load_file(str(directory / "adapter_model.safetensors"))
    expected = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    if set(state) != set(expected):
        raise ValueError(f"Adapter keys differ: missing={set(expected)-set(state)}, unexpected={set(state)-set(expected)}")
    for name in state:
        if state[name].shape != expected[name].shape:
            raise ValueError(f"Adapter shape mismatch: {name}")
    outcome = set_peft_model_state_dict(model, state, adapter_name="default")
    if outcome.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {outcome.unexpected_keys}")
    if not trainable:
        model.requires_grad_(False).eval()
    return model, metadata
