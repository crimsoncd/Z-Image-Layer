"""Two-stage single-layer RGBA training; run from the repository root.

Stage 1:
  python train/train_transparency.py --stage codec --model-dir MODEL \
      --data-dir RGBA_PNG_TXT --output-dir outputs/rgba-codec
Stage 2:
  accelerate launch train/train_transparency.py --stage flow --model-dir MODEL \
      --codec-dir outputs/rgba-codec/checkpoint-10000 --data-dir RGBA_PNG_TXT \
      --output-dir outputs/rgba-flow

Stage 1 freezes the RGB VAE and learns an RGBA latent offset/decoder. Stage 2
freezes that codec and trains attention LoRA with Z-Image's flow time/sign
convention. --init-lora is a weights-only warm start, not optimizer resume.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from train.rgba_data import RGBATextDataset, AspectRatioBatchSampler, encode_captions, parse_buckets
from utils import load_from_local_dir, set_attention_backend
from utils.loader import load_config, load_sharded_safetensors
from zimage.autoencoder import AutoencoderKL
from zimage.transparency import LatentTransparencyVAE, TransparencyConfig, transparency_losses
from zimage.transparency_lora import install_transparency_lora, load_transparency_lora, save_transparency_lora


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("codec", "flow"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, help="Required for flow; optional codec weights warm start.")
    parser.add_argument("--init-lora", type=Path, help="Optional flow LoRA weights warm start.")
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--offset-scale", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--buckets", help="Comma-separated WIDTHxHEIGHT aspect buckets.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, help="Defaults: codec=1e-4, flow=1e-5.")
    parser.add_argument("--mixed-precision", choices=("no", "bf16"), default="bf16")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--caption-dropout", type=float, default=0.1)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--attention-backend", default="native")
    for name, default in (("alpha", 1.0), ("edge", 0.5), ("rgb", 1.0),
                          ("composite", 1.0), ("identity", 1.0), ("offset", 0.01)):
        parser.add_argument(f"--{name}-weight", type=float, default=default)
    args = parser.parse_args()
    for name in ("batch_size", "max_steps", "gradient_accumulation_steps", "save_every", "log_every",
                 "height", "width", "max_sequence_length"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.stage == "flow" and args.codec_dir is None:
        parser.error("--codec-dir is required for flow training")
    if args.stage == "codec" and args.init_lora is not None:
        parser.error("--init-lora applies only to flow training")
    if not 0 <= args.caption_dropout <= 1:
        parser.error("--caption-dropout must be in [0,1]")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.max_grad_norm <= 0 or args.num_workers < 0:
        parser.error("Invalid gradient norm or worker count")
    for name in ("alpha", "edge", "rgb", "composite", "identity", "offset"):
        if getattr(args, f"{name}_weight") < 0:
            parser.error("Loss weights must be nonnegative")
    if args.stage == "codec" and args.alpha_weight + args.rgb_weight + args.composite_weight <= 0:
        parser.error("Codec training requires a positive alpha, RGB or composite reconstruction weight")
    return args


def load_rgb_vae(model_dir, device):
    """Codec training does not need to load the multi-billion-parameter DiT."""
    vae_dir = model_dir / "vae"
    vae = AutoencoderKL(**load_config(str(vae_dir / "config.json")))
    missing, unexpected = vae.load_state_dict(load_sharded_safetensors(vae_dir, device="cpu"), strict=False)
    # This repository's legacy decoder has an extra alpha head. It is frozen
    # and ignored by the transparency wrapper, so original RGB weights suffice.
    if unexpected or any(not key.startswith("decoder.alpha_head.") for key in missing):
        raise ValueError(f"Incompatible base VAE: missing={missing}, unexpected={unexpected}")
    return vae.to(device=device, dtype=torch.float32)


def fp32_context(device):
    return torch.autocast(device_type=device.type, enabled=False) if device.type in ("cuda", "cpu") else nullcontext()


def main():
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import set_seed

    args = parse_args()
    # Different aspect buckets cannot be concatenated into a dispatch batch.
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision if args.stage == "flow" else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False),
    )
    set_seed(args.seed)
    set_attention_backend(args.attention_backend)
    device = accelerator.device
    components = None
    if args.stage == "flow":
        components = load_from_local_dir(args.model_dir, device=str(device),
                                         dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
        base_vae = components["vae"]
    else:
        base_vae = load_rgb_vae(args.model_dir, device)
    codec = (LatentTransparencyVAE.from_pretrained(base_vae, args.codec_dir) if args.codec_dir else
             LatentTransparencyVAE(base_vae, TransparencyConfig(
                 base_vae.config.latent_channels, 2 ** (len(base_vae.config.block_out_channels) - 1),
                 args.hidden_channels, args.offset_scale)))

    buckets = parse_buckets(args.buckets, args.height, args.width)
    patch_size = components["transformer"].all_patch_size[0] if components else 2
    multiple = codec.transparency_config.scale_factor * patch_size
    if any(w < multiple or h < multiple or w % multiple or h % multiple for w, h in buckets):
        raise ValueError(f"Bucket dimensions must be positive multiples of {multiple}.")
    dataset = RGBATextDataset(args.data_dir, buckets)
    loader = DataLoader(dataset, batch_sampler=AspectRatioBatchSampler(dataset.bucket_indices, args.batch_size),
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    lora_config = None
    if args.stage == "flow":
        codec.requires_grad_(False).eval()
        components["text_encoder"].requires_grad_(False).eval()
        model = components["transformer"]
        if model.in_channels != codec.transparency_config.latent_channels:
            raise ValueError("DiT and VAE latent channel counts differ.")
        lora_config = (load_transparency_lora(model, args.init_lora) if args.init_lora else
                       install_transparency_lora(model, args.lora_rank, args.lora_alpha))
    else:
        model = codec
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate or (1e-4 if args.stage == "codec" else 1e-5))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if len(loader) == 0:
        raise ValueError("Training loader is empty on this process.")
    model.train()
    step = 0

    def save(step):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            directory = args.output_dir / f"checkpoint-{step}"
            raw_model = accelerator.unwrap_model(model)
            (raw_model if args.stage == "codec" else codec).save_pretrained(directory)
            if args.stage == "flow":
                save_transparency_lora(raw_model, lora_config, directory)
            config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            config["global_step"] = step
            (directory / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        accelerator.wait_for_everyone()

    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    while step < args.max_steps:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        for batch in loader:
            rgb = batch["rgb"].to(device=device, dtype=torch.float32)
            alpha = batch["alpha"].to(device=device, dtype=torch.float32)
            with accelerator.accumulate(model):
                if args.stage == "codec":
                    outputs = model(rgb, alpha)
                    losses = transparency_losses(outputs, rgb, alpha)
                    loss = sum(getattr(args, f"{name}_weight") * value for name, value in losses.items())
                else:
                    with torch.no_grad():
                        with fp32_context(device):
                            clean = codec.to_diffusion(codec.encode_rgba(rgb, alpha))
                        captions = encode_captions(list(batch["caption"]), components["tokenizer"],
                                                   components["text_encoder"], device,
                                                   args.max_sequence_length, args.caption_dropout)
                    # Continuous logit-normal flow sampling: sigma=1 is noise,
                    # model time=1-sigma; pipeline negates the predicted velocity.
                    sigma = torch.randn(clean.shape[0], device=device).sigmoid().view(-1, 1, 1, 1)
                    noise = torch.randn_like(clean)
                    noisy = (1 - sigma) * clean + sigma * noise
                    model_time = 1 - sigma.flatten()
                    model_dtype = next(accelerator.unwrap_model(model).parameters()).dtype
                    with accelerator.autocast():
                        prediction = model(list(noisy.to(model_dtype).unsqueeze(2).unbind(0)), model_time, captions)[0]
                        prediction = torch.stack(prediction).squeeze(2).float()
                        loss = F.mse_loss(prediction, clean - noise)
                    losses = {"flow": loss}
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                if step % args.log_every == 0:
                    accelerator.print(f"step={step}/{args.max_steps} " + " ".join(
                        f"{name}={value.detach().item():.5f}" for name, value in losses.items()))
                if step % args.save_every == 0:
                    save(step)
                if step >= args.max_steps:
                    break
        epoch += 1
    if step % args.save_every:
        save(step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
