"""Fine-tune Z-Image and its VAE alpha head on paired RGBA/text data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import load_from_local_dir, set_attention_backend  # noqa: E402


class RGBATextDataset(Dataset):
    """Directory dataset containing matching ``name.png``/``name.txt`` pairs."""

    def __init__(
        self,
        root: str | Path,
        height: int,
        width: int,
        random_crop: bool = True,
        horizontal_flip: bool = True,
    ) -> None:
        self.root = Path(root)
        self.height = height
        self.width = width
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip

        images = sorted(path for path in self.root.iterdir() if path.is_file() and path.suffix.lower() == ".png")
        self.samples = [(path, path.with_suffix(".txt")) for path in images if path.with_suffix(".txt").is_file()]
        if not self.samples:
            raise ValueError(f"No matching PNG/TXT pairs found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_and_crop(self, image: Image.Image) -> Image.Image:
        scale = max(self.width / image.width, self.height / image.height)
        resized_width = max(self.width, round(image.width * scale))
        resized_height = max(self.height, round(image.height * scale))
        image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        max_left = resized_width - self.width
        max_top = resized_height - self.height
        if self.random_crop:
            left = random.randint(0, max_left) if max_left else 0
            top = random.randint(0, max_top) if max_top else 0
        else:
            left = max_left // 2
            top = max_top // 2
        return image.crop((left, top, left + self.width, top + self.height))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        image_path, text_path = self.samples[index]
        with Image.open(image_path) as source:
            image = self._resize_and_crop(source.convert("RGBA"))

        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        rgba = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        rgb, alpha = rgba[:3], rgba[3:4]

        # Resampling can introduce tiny RGB values into fully transparent pixels.
        # Restore the dataset invariant while retaining colors on soft edges.
        rgb = rgb * (alpha > 0).to(rgb.dtype)
        caption = text_path.read_text(encoding="utf-8").strip()
        return {"rgb": rgb, "alpha": alpha, "caption": caption}


class DecoderTrainingWrapper(nn.Module):
    """Expose VAE decoding through ``forward`` so distributed wrappers sync it."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alpha-training"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint directory containing trainable weights.")

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--no-horizontal-flip", action="store_true")

    parser.add_argument("--transformer-train-mode", choices=("full", "last", "none"), default="full")
    parser.add_argument("--transformer-last-layers", type=int, default=4)
    parser.add_argument("--decoder-train-mode", choices=("alpha", "late", "full"), default="alpha")
    parser.add_argument("--transformer-lr", type=float, default=1e-6)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--caption-dropout", type=float, default=0.1)
    parser.add_argument("--posterior-sampling", action="store_true")

    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--alpha-bce-weight", type=float, default=1.0)
    parser.add_argument("--alpha-dice-weight", type=float, default=0.5)
    parser.add_argument("--alpha-edge-weight", type=float, default=0.25)
    parser.add_argument("--composite-weight", type=float, default=0.5)
    parser.add_argument("--rgb-weight", type=float, default=0.1)

    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--attention-backend", default="_native_flash")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def set_trainable_parameters(
    transformer: nn.Module,
    vae: nn.Module,
    transformer_mode: str,
    decoder_mode: str,
    last_layers: int,
) -> None:
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if transformer_mode == "full":
        transformer.requires_grad_(True)
    elif transformer_mode == "last":
        if last_layers < 1 or last_layers > len(transformer.layers):
            raise ValueError(f"--transformer-last-layers must be within [1, {len(transformer.layers)}]")
        for layer in transformer.layers[-last_layers:]:
            layer.requires_grad_(True)
        transformer.all_final_layer.requires_grad_(True)

    vae.decoder.alpha_head.requires_grad_(True)
    if decoder_mode in ("late", "full"):
        vae.decoder.conv_norm_out.requires_grad_(True)
        vae.decoder.conv_out.requires_grad_(True)
        vae.decoder.up_blocks[-1].requires_grad_(True)
    if decoder_mode == "full":
        vae.decoder.requires_grad_(True)
        if vae.post_quant_conv is not None:
            vae.post_quant_conv.requires_grad_(True)


def vae_encode(vae: nn.Module, rgb: torch.Tensor, sample_posterior: bool) -> torch.Tensor:
    """Encode RGB in [0, 1] using the frozen VAE posterior."""
    moments = vae.encoder(rgb.mul(2).sub(1))
    if vae.quant_conv is not None:
        moments = vae.quant_conv(moments)
    mean, logvar = moments.chunk(2, dim=1)
    if sample_posterior:
        latent = mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * torch.randn_like(mean)
    else:
        latent = mean

    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    return (latent - shift) * vae.config.scaling_factor


def encode_captions(
    captions: List[str],
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device,
    max_length: int,
    dropout: float,
) -> List[torch.Tensor]:
    captions = ["" if random.random() < dropout else caption for caption in captions]
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": caption}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for caption in captions
    ]
    tokens = tokenizer(
        formatted,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device).bool()
    hidden = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-2]
    return [hidden[i][attention_mask[i]].detach() for i in range(len(hidden))]


def sample_flow_batch(clean: torch.Tensor, scheduler) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct x(t) and target using the sign/time convention in pipeline.py."""
    indices = torch.randint(0, len(scheduler.sigmas), (clean.shape[0],), device=clean.device)
    sigma = scheduler.sigmas.to(device=clean.device)[indices].view(-1, 1, 1, 1)
    noise = torch.randn_like(clean)
    noisy = (1.0 - sigma) * clean + sigma * noise
    model_time = (1.0 - sigma.flatten()).to(clean.dtype)
    target = clean - noise
    return noisy, model_time, target


def balanced_alpha_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Balance foreground/background without changing the optimum calibration."""
    occupancy = (target >= 0.5).to(target.dtype)
    foreground = occupancy.mean(dim=(1, 2, 3), keepdim=True).clamp(1e-3, 1.0 - 1e-3)
    weights = occupancy * (0.5 / foreground) + (1.0 - occupancy) * (0.5 / (1.0 - foreground))
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()


def dice_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def reconstruction_losses(decoded: torch.Tensor, rgb: torch.Tensor, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred_rgb = decoded[:, :3].div(2).add(0.5).clamp(0.0, 1.0)
    alpha_logits = decoded[:, 3:4]
    pred_alpha = alpha_logits.sigmoid()

    background = torch.rand((rgb.shape[0], 3, 1, 1), device=rgb.device, dtype=rgb.dtype)
    pred_composite = pred_rgb * pred_alpha + background * (1.0 - pred_alpha)
    target_composite = rgb * alpha + background * (1.0 - alpha)
    foreground_weight = 0.25 + 0.75 * alpha

    return {
        "alpha_bce": balanced_alpha_bce(alpha_logits, alpha),
        "alpha_dice": dice_loss(pred_alpha, alpha),
        "alpha_edge": edge_loss(pred_alpha, alpha),
        "composite": F.l1_loss(pred_composite, target_composite),
        "rgb": (torch.abs(pred_rgb - rgb) * foreground_weight).mean(),
    }


def cosine_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def trainable_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
        if name in trainable_names
    }


def load_trainable_weights(module: nn.Module, path: Path) -> None:
    if not path.is_file():
        return
    missing, unexpected = module.load_state_dict(load_file(str(path)), strict=False)
    if unexpected:
        raise ValueError(f"Unexpected weights in {path}: {unexpected}")


def save_checkpoint(accelerator, transformer: nn.Module, decoder: nn.Module, output_dir: Path, step: int) -> None:
    if not accelerator.is_main_process:
        return
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_transformer = accelerator.unwrap_model(transformer)
    raw_decoder = accelerator.unwrap_model(decoder)
    transformer_state = trainable_state_dict(raw_transformer)
    decoder_state = trainable_state_dict(raw_decoder)
    if transformer_state:
        save_file(transformer_state, str(checkpoint_dir / "transformer_trainable.safetensors"))
    save_file(decoder_state, str(checkpoint_dir / "vae_trainable.safetensors"))
    (checkpoint_dir / "training_state.json").write_text(json.dumps({"global_step": step}, indent=2), encoding="utf-8")


def parameter_groups(
    transformer: nn.Module,
    decoder: nn.Module,
    transformer_lr: float,
    decoder_lr: float,
) -> List[Dict]:
    groups = []
    transformer_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    decoder_parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    if transformer_parameters:
        groups.append({"params": transformer_parameters, "lr": transformer_lr})
    if decoder_parameters:
        groups.append({"params": decoder_parameters, "lr": decoder_lr})
    return groups


def main() -> None:
    args = parse_args()
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
    except ImportError as error:
        raise RuntimeError("Install the project dependencies, including accelerate, before training.") from error

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)
    set_attention_backend(args.attention_backend)

    dataset = RGBATextDataset(
        args.data_dir,
        args.height,
        args.width,
        random_crop=not args.no_random_crop,
        horizontal_flip=not args.no_horizontal_flip,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model_dtype = torch.float32
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16
    components = load_from_local_dir(
        args.model_dir,
        device=str(accelerator.device),
        dtype=model_dtype,
        verbose=accelerator.is_main_process,
    )
    transformer = components["transformer"]
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    noise_scheduler = components["scheduler"]

    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    required_multiple = vae_scale * transformer.all_patch_size[0]
    if args.height % required_multiple or args.width % required_multiple:
        raise ValueError(
            f"Training height and width must be divisible by {required_multiple}; "
            f"got {args.height}x{args.width}."
        )

    set_trainable_parameters(
        transformer,
        vae,
        args.transformer_train_mode,
        args.decoder_train_mode,
        args.transformer_last_layers,
    )
    decoder = DecoderTrainingWrapper(vae)
    text_encoder.requires_grad_(False).eval()
    vae.encoder.eval()

    if args.resume is not None:
        load_trainable_weights(transformer, args.resume / "transformer_trainable.safetensors")
        load_trainable_weights(decoder, args.resume / "vae_trainable.safetensors")

    groups = parameter_groups(transformer, decoder, args.transformer_lr, args.decoder_lr)
    if not groups:
        raise ValueError("No trainable parameters were selected.")
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps or args.epochs * updates_per_epoch
    lr_scheduler = cosine_schedule(optimizer, args.warmup_steps, total_steps)

    if args.transformer_train_mode == "none":
        decoder, optimizer, dataloader = accelerator.prepare(decoder, optimizer, dataloader)
    else:
        transformer, decoder, optimizer, dataloader = accelerator.prepare(
            transformer, decoder, optimizer, dataloader
        )
    raw_vae = accelerator.unwrap_model(decoder).vae
    transformer.train(args.transformer_train_mode != "none")
    decoder.train()
    raw_vae.encoder.eval()

    global_step = 0
    if args.resume is not None and (args.resume / "training_state.json").is_file():
        state = json.loads((args.resume / "training_state.json").read_text(encoding="utf-8"))
        global_step = int(state.get("global_step", 0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch in dataloader:
            rgb = batch["rgb"].to(accelerator.device, dtype=torch.float32, non_blocking=True)
            alpha = batch["alpha"].to(accelerator.device, dtype=torch.float32, non_blocking=True)

            with torch.no_grad():
                clean_latents = vae_encode(raw_vae, rgb, args.posterior_sampling)
                caption_features = encode_captions(
                    list(batch["caption"]),
                    tokenizer,
                    text_encoder,
                    accelerator.device,
                    args.max_sequence_length,
                    args.caption_dropout,
                )

            with accelerator.accumulate(transformer, decoder):
                losses: Dict[str, torch.Tensor] = {}
                with accelerator.autocast():
                    if args.transformer_train_mode != "none":
                        noisy, model_time, flow_target = sample_flow_batch(clean_latents, noise_scheduler)
                        transformer_dtype = next(transformer.parameters()).dtype
                        noisy_items = list(noisy.to(transformer_dtype).unsqueeze(2).unbind(dim=0))
                        predicted_items = transformer(noisy_items, model_time, caption_features)[0]
                        predicted_flow = torch.stack(predicted_items, dim=0).squeeze(2).float()
                        losses["flow"] = F.mse_loss(predicted_flow, flow_target.float())
                    else:
                        losses["flow"] = torch.zeros((), device=accelerator.device)

                    decoded = decoder(clean_latents.float())
                    losses.update(reconstruction_losses(decoded.float(), rgb, alpha))
                    total_loss = (
                        args.flow_loss_weight * losses["flow"]
                        + args.alpha_bce_weight * losses["alpha_bce"]
                        + args.alpha_dice_weight * losses["alpha_dice"]
                        + args.alpha_edge_weight * losses["alpha_edge"]
                        + args.composite_weight * losses["composite"]
                        + args.rgb_weight * losses["rgb"]
                    )

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    trainable_parameters: Iterable[torch.Tensor] = (
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    )
                    accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % args.log_every == 0:
                    summary = " ".join(f"{name}={value.detach().item():.4f}" for name, value in losses.items())
                    print(f"step={global_step}/{total_steps} loss={total_loss.detach().item():.4f} {summary}", flush=True)
                if global_step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    save_checkpoint(accelerator, transformer, decoder, args.output_dir, global_step)
                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, transformer, decoder, args.output_dir, global_step)
    if accelerator.is_main_process:
        print(f"Training complete. Final checkpoint: {args.output_dir / f'checkpoint-{global_step}'}")


if __name__ == "__main__":
    main()
