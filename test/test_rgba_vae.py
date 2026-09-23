"""Inspect a trained codec by reconstructing an RGBA image (no DiT needed).

python reconstruct_rgba.py --model-dir MODEL --checkpoint CODEC_CHECKPOINT \
    --image example.png --output-dir outputs/reconstruction

Native resolution is retained unless --max-side is given. Images are padded
on the right/bottom to the VAE scale factor, then cropped back after decoding.
"""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

# sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, "/remote-home/Zhangkaile/dev/Z-Image-Layer/src")

from utils.loader import load_config, load_sharded_safetensors
from zimage.autoencoder import AutoencoderKL
from zimage.transparency import LatentTransparencyVAE


def to_pil(tensor):
    values = tensor.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
    channels, height, width = values.shape
    mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]
    return Image.frombytes(mode, (width, height), values.permute(1, 2, 0).contiguous().numpy().tobytes())


def checkerboard(height, width, cell=24):
    rows = torch.arange(height)[:, None] // cell
    cols = torch.arange(width)[None, :] // cell
    return (0.65 + 0.2 * ((rows + cols) % 2)).float().unsqueeze(0).expand(3, -1, -1)


def compose(rgb, alpha, background):
    return rgb * alpha + background * (1 - alpha)


def save_comparison(rows, path):
    width, height = rows[0][0][1].size
    label_height, gap = 28, 8
    canvas = Image.new("RGB", (3 * width + 4 * gap, len(rows) * (height + label_height + gap) + gap), "#202020")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        for column, (label, panel) in enumerate(row):
            left = gap + column * (width + gap)
            top = gap + row_index * (height + label_height + gap)
            draw.text((left + 4, top + 6), label, fill="white")
            canvas.paste(panel.convert("RGB"), (left, top + label_height))
    canvas.save(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("test/reconstruction"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int, default=512, help="Optionally shrink the longest side; never upscale.")
    parser.add_argument("--error-gain", type=float, default=5.0, help="Fixed amplification for error panels only.")
    args = parser.parse_args()
    if args.max_side is not None and args.max_side < 1:
        parser.error("--max-side must be positive")
    if args.error_gain <= 0:
        parser.error("--error-gain must be positive")

    with Image.open(args.image) as source:
        if "A" not in source.getbands() and "transparency" not in source.info:
            raise ValueError("Input must contain a real alpha channel.")
        image = source.convert("RGBA")
    original_size = image.size
    if args.max_side and max(image.size) > args.max_side:
        scale = args.max_side / max(image.size)
        size = tuple(max(1, round(value * scale)) for value in image.size)
        image = image.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    width, height = image.size
    rgba = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    rgba = rgba.reshape(height, width, 4).permute(2, 0, 1).float().div(255)

    vae_dir = args.model_dir / "vae"
    base = AutoencoderKL(**load_config(str(vae_dir / "config.json")))
    missing, unexpected = base.load_state_dict(load_sharded_safetensors(vae_dir, device="cpu"), strict=False)
    if unexpected or any(not key.startswith("decoder.alpha_head.") for key in missing):
        raise ValueError(f"Incompatible base VAE: missing={missing}, unexpected={unexpected}")
    base.to(device=args.device, dtype=torch.float32)
    codec = LatentTransparencyVAE.from_pretrained(base, args.checkpoint).requires_grad_(False).eval()

    factor = codec.transparency_config.scale_factor
    padded = F.pad(rgba.unsqueeze(0), (0, (-width) % factor, 0, (-height) % factor)).to(args.device)
    # These APIs both use raw VAE latents. Do not apply diffusion scaling here.
    latent = codec.encode_rgba(padded[:, :3], padded[:, 3:4])
    decoded = codec.decode(latent, return_dict=False)[0][0, :, :height, :width].float().cpu()
    if not torch.isfinite(decoded).all():
        raise RuntimeError("Codec produced non-finite outputs.")
    predicted_rgb = (decoded[:3] / 2 + 0.5).clamp(0, 1)
    predicted_alpha = decoded[3:4].sigmoid()
    rgb, alpha = rgba[:3], rgba[3:4]
    reconstruction = torch.cat((predicted_rgb, predicted_alpha))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid overwriting the input when it happens to be named reference.png, etc.
    names = ("reference.png", "reconstruction.png", "alpha_reference.png", "alpha_reconstruction.png",
             "alpha_error.png", "comparison.png", "metrics.json")
    if args.image.resolve() in {(args.output_dir / name).resolve() for name in names}:
        raise ValueError("Choose an output directory different from the input image's directory.")
    to_pil(rgba).save(args.output_dir / "reference.png")
    to_pil(reconstruction).save(args.output_dir / "reconstruction.png")
    to_pil(alpha).save(args.output_dir / "alpha_reference.png")
    to_pil(predicted_alpha).save(args.output_dir / "alpha_reconstruction.png")
    alpha_error = (predicted_alpha - alpha).abs()
    to_pil(alpha_error * args.error_gain).save(args.output_dir / "alpha_error.png")
    metrics = {
        "source_size_wh": list(original_size), "evaluated_size_wh": [width, height],
        "alpha_mae": alpha_error.mean().item(),
        "alpha_mse": alpha_error.square().mean().item(),
        "visible_rgb_mae": (((predicted_rgb - rgb).abs() * alpha).sum() / (3 * alpha.sum())).item()
        if alpha.sum() > 0 else None,
        "error_display_gain": args.error_gain,
    }
    rows = []
    for name, background in (("Checker", checkerboard(height, width)),
                             ("White", torch.ones_like(rgb)), ("Black", torch.zeros_like(rgb))):
        reference = compose(rgb, alpha, background)
        prediction = compose(predicted_rgb, predicted_alpha, background)
        error = (prediction - reference).abs()
        metrics[f"composite_{name.lower()}_mae"] = error.mean().item()
        rows.append([(f"{name}: reference", to_pil(reference)),
                     (f"{name}: reconstruction", to_pil(prediction)),
                     (f"Absolute error x{args.error_gain:g}", to_pil(error * args.error_gain))])
    rows.append([("Alpha: reference", to_pil(alpha)), ("Alpha: reconstruction", to_pil(predicted_alpha)),
                 (f"Alpha error x{args.error_gain:g}", to_pil(alpha_error * args.error_gain))])
    save_comparison(rows, args.output_dir / "comparison.png")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved reconstruction and comparison to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
