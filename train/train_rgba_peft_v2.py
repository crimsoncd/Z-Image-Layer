"""Feature-complete stage-2 RGBA flow training using PEFT's low-level API.

This is a compatible extension of train_rgba_peft.py. It supports mixing
multiple dataset roots and resuming adapter/optimizer state from a checkpoint.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from train.rgba_data import RGBATextDataset, AspectRatioBatchSampler, parse_buckets, encode_captions
from utils import load_from_local_dir, set_attention_backend
from zimage.transparency import LatentTransparencyVAE
from zimage.peft_rgba import resolve_targets, create_lora, load_adapter, save_adapter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--codec-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, nargs="+", help="One or more dataset directories.")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--init-adapter", type=Path, help="PEFT weights-only warm start; step starts at zero.")
    source.add_argument("--resume-from-checkpoint", type=Path,
                        help="Resume adapter, global step, and optimizer state when available.")
    p.add_argument("--groups", default="attention,mlp")
    p.add_argument("--scopes", default="layers,noise_refiner")
    p.add_argument("--blocks", help="Main layers only, zero-based inclusive ranges: 0-3,20,25-29")
    p.add_argument("--target-regex", help="Full module-name regex; replaces groups/scopes/blocks.")
    p.add_argument("--exclude-regex", help="Regex search on selected module names to exclude.")
    p.add_argument("--list-targets", action="store_true", help="Print selection and exit without training/saving.")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=float, default=64)
    p.add_argument("--lora-dropout", type=float, default=0)
    p.add_argument("--use-dora", action="store_true")
    p.add_argument("--use-rslora", action="store_true")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--buckets", default="512x512")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10000,
                   help="Global target update count, including completed resume steps.")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--save-steps", default="1", help="Extra global update numbers to save, e.g. 1,100,500")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1)
    p.add_argument("--mixed-precision", choices=("no", "bf16"), default="bf16")
    p.add_argument("--caption-dropout", type=float, default=0.1)
    p.add_argument("--caption-suffix", default="", help="Added in memory; original TXT files remain unchanged.")
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention-backend", default="native")
    args = p.parse_args()
    if args.max_steps < 0 or (args.max_steps > 0 and args.data_dir is None and not args.list_targets):
        p.error("max-steps must be >=0; training requires --data-dir")
    for field in ("batch_size", "gradient_accumulation_steps", "save_every", "log_every", "height", "width",
                  "rank", "max_sequence_length"):
        if getattr(args, field) < 1:
            p.error(f"{field} must be positive")
    if args.num_workers < 0 or args.learning_rate <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        p.error("Invalid worker count or optimizer settings")
    if not 0 <= args.caption_dropout <= 1 or not 0 <= args.lora_dropout < 1 or args.lora_alpha <= 0:
        p.error("Invalid caption/LoRA dropout or alpha")
    return args


def printable_arguments(args):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return {name: convert(value) for name, value in vars(args).items()}


def combined_dataset(roots, buckets):
    datasets = [RGBATextDataset(root, buckets) for root in roots]
    dataset = ConcatDataset(datasets)
    bucket_indices = [[] for _ in buckets]
    offset = 0
    for child in datasets:
        for bucket, indices in enumerate(child.bucket_indices):
            bucket_indices[bucket].extend(offset + index for index in indices)
        offset += len(child)
    return dataset, bucket_indices, [len(child) for child in datasets]


def main():
    args = parse_args()
    extra_steps = {int(item) for item in args.save_steps.split(",") if item.strip()}
    if any(step < 1 for step in extra_steps):
        raise ValueError("--save-steps must contain positive update numbers.")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    accelerator.print("Training configuration:\n" + json.dumps(printable_arguments(args), indent=2, ensure_ascii=False))
    device = accelerator.device
    set_seed(args.seed)
    set_attention_backend(args.attention_backend)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.list_targets or args.resume_from_checkpoint):
        raise FileExistsError(f"Use a new empty output directory: {args.output_dir}")
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    components = load_from_local_dir(args.model_dir, device=str(device), dtype=dtype)
    codec = LatentTransparencyVAE.from_pretrained(components["vae"], args.codec_dir).requires_grad_(False).eval()
    text_encoder = components["text_encoder"].requires_grad_(False).eval()
    model = components["transformer"]
    if model.in_channels != codec.transparency_config.latent_channels:
        raise ValueError("VAE and transformer latent channels do not match.")
    if 2 not in model.all_patch_size or 1 not in model.all_f_patch_size:
        raise ValueError("This trainer uses the native pipeline's patch_size=2, f_patch_size=1.")

    adapter_source = args.resume_from_checkpoint or args.init_adapter
    if adapter_source:
        model, parent_metadata = load_adapter(model, adapter_source, trainable=True)
        targets = parent_metadata["targets"]
        accelerator.print("Adapter config/targets loaded from checkpoint; selection flags are ignored.")
    else:
        targets = resolve_targets(model, args.groups, args.scopes, args.blocks, args.target_regex, args.exclude_regex)
        parent_metadata = None
        if not args.list_targets:
            model = create_lora(model, targets, args.rank, args.lora_alpha, args.lora_dropout,
                                args.use_dora, args.use_rslora)
    accelerator.print("Selected LoRA modules:\n" + "\n".join(targets))
    if args.list_targets:
        accelerator.end_training()
        return

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    accelerator.print(f"Trainable parameters: {sum(parameter.numel() for parameter in parameters):,}")
    if not parameters:
        raise ValueError("No trainable adapter parameters")
    step = int(parent_metadata.get("step", 0)) if args.resume_from_checkpoint else 0
    if step < 0 or step > args.max_steps:
        raise ValueError(f"Checkpoint step {step} is outside requested max-steps {args.max_steps}")
    metadata = {
        "targets": targets,
        "base_model_dir": str(args.model_dir),
        "codec_source": str(args.codec_dir),
        "warm_start": str(args.init_adapter) if args.init_adapter else None,
        "resumed_from": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
        "parent_step": parent_metadata.get("step") if parent_metadata else None,
        "arguments": printable_arguments(args),
    }

    optimizer = None
    epoch = 0
    loader = None
    dataset = None
    if args.max_steps > step:
        buckets = parse_buckets(args.buckets, args.height, args.width)
        multiple = codec.transparency_config.scale_factor * 2
        if any(w < multiple or h < multiple or w % multiple or h % multiple for w, h in buckets):
            raise ValueError(f"Buckets must be positive multiples of {multiple}")
        dataset, bucket_indices, dataset_sizes = combined_dataset(args.data_dir, buckets)
        loader = DataLoader(dataset, batch_sampler=AspectRatioBatchSampler(bucket_indices, args.batch_size),
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        state_path = args.resume_from_checkpoint / "training_state.pt" if args.resume_from_checkpoint else None
        if state_path and state_path.is_file():
            training_state = torch.load(state_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(training_state["optimizer"])
            epoch = int(training_state.get("epoch", 0))
            accelerator.print(f"Restored optimizer state from {state_path}")
        elif args.resume_from_checkpoint:
            accelerator.print("Checkpoint has no training_state.pt; optimizer starts fresh.")
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
        if len(loader) == 0:
            raise ValueError("Empty training loader")
        accelerator.print(f"dataset_sizes={dataset_sizes} total_samples={len(dataset)} "
                          f"effective_batch={args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")

    def save(save_step):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            directory = args.output_dir / f"checkpoint-{save_step}"
            directory.mkdir(parents=True, exist_ok=False)
            raw = accelerator.unwrap_model(model)
            codec.save_pretrained(directory)
            save_adapter(raw, directory, {**metadata, "step": save_step,
                                         "untrained_adapter": save_step == 0 and adapter_source is None})
            if optimizer is not None:
                torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch}, directory / "training_state.pt")
            (directory / "COMPLETE").write_text("Checkpoint saved successfully.\n", encoding="utf-8")
            print(f"Saved {directory}", flush=True)
        accelerator.wait_for_everyone()

    if not args.resume_from_checkpoint:
        save(0)
    if args.max_steps == step:
        accelerator.print(f"Already at target step {step}; no updates required.")
        accelerator.end_training()
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    while step < args.max_steps:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        for batch in loader:
            rgb = batch["rgb"].to(device=device, dtype=torch.float32)
            alpha = batch["alpha"].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                context = torch.autocast(device.type, enabled=False) if device.type in ("cuda", "cpu") else nullcontext()
                with context:
                    clean = codec.to_diffusion(codec.encode_rgba(rgb, alpha))
                captions = [(caption + " " + args.caption_suffix).strip() for caption in batch["caption"]]
                features = encode_captions(captions, components["tokenizer"], text_encoder, device,
                                           args.max_sequence_length, args.caption_dropout)
            with accelerator.accumulate(model):
                sigma = torch.randn(clean.shape[0], device=device).sigmoid().view(-1, 1, 1, 1)
                noise = torch.randn_like(clean)
                noisy = (1 - sigma) * clean + sigma * noise
                with accelerator.autocast():
                    predicted = model(list(noisy.to(dtype).unsqueeze(2).unbind(0)), 1 - sigma.flatten(), features)[0]
                    predicted = torch.stack(predicted).squeeze(2).float()
                    losses = {"mse": F.mse_loss(predicted, clean - noise)}
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at update {step}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not any(parameter.grad is not None for parameter in parameters):
                        raise RuntimeError("No adapter gradients: check selected targets")
                    grad_norm = accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                totals[0] += loss.detach().double() * rgb.shape[0]
                totals[1] += rgb.shape[0]
            if accelerator.sync_gradients:
                step += 1
                if step % args.log_every == 0 or step == args.max_steps:
                    accelerator.print(f"step={step}/{args.max_steps} loss={loss.detach().item():.5f} "
                                      f"lr={optimizer.param_groups[0]['lr']:.6g} " + " ".join(
                                          f"{name}={value.detach().item():.5f}"
                                          for name, value in losses.items()))
                    sums = accelerator.reduce(totals, reduction="sum")
                    metric = {"step": step, "loss_mean": (sums[0] / sums[1]).item(),
                              "loss_last": loss.detach().item(), "grad_norm": float(grad_norm),
                              "lr": optimizer.param_groups[0]["lr"], "samples_in_interval": int(sums[1].item())}
                    if accelerator.is_main_process:
                        args.output_dir.mkdir(parents=True, exist_ok=True)
                        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(metric) + "\n")
                    totals.zero_()
                if step in extra_steps or step % args.save_every == 0 or step == args.max_steps:
                    save(step)
                if step >= args.max_steps:
                    break
        epoch += 1
    accelerator.end_training()


if __name__ == "__main__":
    main()
