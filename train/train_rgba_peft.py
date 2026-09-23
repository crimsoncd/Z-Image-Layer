"""Independent stage-2 RGBA flow training using PEFT's low-level API.

See train/README_PEFT.md. Every fresh run saves checkpoint-0 before updates.
--max-steps 0 exports an untrained adapter without reading a dataset.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
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
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--init-adapter", type=Path, help="PEFT weights-only warm start; not optimizer resume.")
    p.add_argument("--groups", default="attention,mlp")
    p.add_argument("--scopes", default="layers,noise_refiner")
    p.add_argument("--blocks", help="Main layers only, zero-based inclusive ranges: 0-3,20,25-29")
    p.add_argument("--target-regex", help="Full module-name regex; replaces groups/scopes/blocks.")
    p.add_argument("--exclude-regex", help="Regex search on selected module names to exclude.")
    p.add_argument("--list-targets", action="store_true", help="Print selection and exit without training/saving.")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=32)
    p.add_argument("--lora-dropout", type=float, default=0)
    p.add_argument("--use-dora", action="store_true")
    p.add_argument("--use-rslora", action="store_true")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--buckets", default="512x512")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--save-steps", default="1", help="Extra update numbers to save, e.g. 1,100,500")
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


def main():
    args = parse_args()
    extra_steps = {int(item) for item in args.save_steps.split(",") if item.strip()}
    if any(step < 1 for step in extra_steps):
        raise ValueError("--save-steps must contain positive update numbers; step 0 is saved automatically.")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device
    set_seed(args.seed)
    set_attention_backend(args.attention_backend)
    # Refuse to overwrite previous experiments, including an incomplete step-0.
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.list_targets:
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
    if args.init_adapter:
        model, parent_metadata = load_adapter(model, args.init_adapter, trainable=True)
        targets = parent_metadata["targets"]
        accelerator.print("Warm start: adapter config/targets come from the checkpoint; selection flags are ignored.")
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
    parameters = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(f"Trainable parameters: {sum(p.numel() for p in parameters):,}")
    if not parameters:
        raise ValueError("No trainable adapter parameters")
    metadata = {"targets": targets, "base_model_dir": str(args.model_dir), "codec_source": str(args.codec_dir),
                "warm_start": str(args.init_adapter) if args.init_adapter else None,
                "parent_step": parent_metadata.get("step") if parent_metadata else None,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}

    def save(step):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            directory = args.output_dir / f"checkpoint-{step}"
            directory.mkdir(parents=True, exist_ok=False)
            raw = accelerator.unwrap_model(model)
            codec.save_pretrained(directory)
            save_adapter(raw, directory, {**metadata, "step": step,
                                         "untrained_adapter": step == 0 and args.init_adapter is None})
            (directory / "COMPLETE").write_text("Checkpoint saved successfully.\n", encoding="utf-8")
            print(f"Saved {directory}", flush=True)
        accelerator.wait_for_everyone()

    # Saving does not consume RNG, so the first update is identical with/without
    # later checkpoint comparisons. Step 0 contains a genuine zero-B adapter.
    save(0)
    if args.max_steps == 0:
        accelerator.end_training()
        return
    buckets = parse_buckets(args.buckets, args.height, args.width)
    multiple = codec.transparency_config.scale_factor * 2
    if any(w < multiple or h < multiple or w % multiple or h % multiple for w, h in buckets):
        raise ValueError(f"Buckets must be positive multiples of {multiple}")
    dataset = RGBATextDataset(args.data_dir, buckets)
    loader = DataLoader(dataset, batch_sampler=AspectRatioBatchSampler(dataset.bucket_indices, args.batch_size),
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if len(loader) == 0:
        raise ValueError("Empty training loader")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step, epoch = 0, 0
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    accelerator.print(f"samples={len(dataset)} effective_batch={args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
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
                    loss = F.mse_loss(predicted, clean - noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at update {step}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not any(p.grad is not None for p in parameters):
                        raise RuntimeError("No adapter gradients: check selected targets")
                    grad_norm = accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                totals[0] += loss.detach().double() * rgb.shape[0]
                totals[1] += rgb.shape[0]
            if accelerator.sync_gradients:
                step += 1
                if step % args.log_every == 0 or step == args.max_steps:
                    sums = accelerator.reduce(totals, reduction="sum")
                    mean_loss = (sums[0] / sums[1]).item()
                    metric = {"step": step, "loss_mean": mean_loss, "loss_last": loss.detach().item(),
                              "grad_norm": float(grad_norm), "lr": args.learning_rate,
                              "samples_in_interval": int(sums[1].item())}
                    accelerator.print(json.dumps(metric), flush=True)
                    if accelerator.is_main_process:
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
