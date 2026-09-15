"""Generate a transparent PNG with a trained latent-transparency checkpoint.

python inference_rgba.py --model-dir MODEL --checkpoint outputs/rgba-flow/checkpoint-10000 \
    --prompt "a glass vase with flowers" --output outputs/vase.png

The checkpoint must contain both codec and flow LoRA weights from stage 2.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils import load_from_local_dir, set_attention_backend
from zimage.pipeline import generate
from zimage.transparency import LatentTransparencyVAE
from zimage.transparency_lora import load_transparency_lora


def load_rgba_components(model_dir, checkpoint, device="cuda", dtype=torch.bfloat16):
    checkpoint = Path(checkpoint)
    required = ("transparency_config.json", "transparency.safetensors", "lora_config.json", "transformer_lora.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"A stage-2 RGBA checkpoint is required; missing: {missing}")
    components = load_from_local_dir(model_dir, device=device, dtype=dtype)
    codec = LatentTransparencyVAE.from_pretrained(components["vae"], checkpoint)
    if codec.transparency_config.latent_channels != components["transformer"].in_channels:
        raise ValueError("DiT and transparency VAE latent channel counts differ.")
    load_transparency_lora(components["transformer"], checkpoint)
    components["vae"] = codec.requires_grad_(False).eval()
    components["transformer"].requires_grad_(False).eval()
    return components


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--output", type=Path, default=Path("outputs/rgba.png"))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--attention-backend", default="native")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png to preserve RGBA")
    if min(args.height, args.width, args.steps) < 1:
        parser.error("Image dimensions and sampling steps must be positive")
    set_attention_backend(args.attention_backend)
    components = load_rgba_components(args.model_dir, args.checkpoint, args.device,
                                      torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    images = generate(**components, prompt=args.prompt, negative_prompt=args.negative_prompt,
                      height=args.height, width=args.width, num_inference_steps=args.steps,
                      guidance_scale=args.guidance_scale,
                      generator=torch.Generator(device=args.device).manual_seed(args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.output)


if __name__ == "__main__":
    main()
