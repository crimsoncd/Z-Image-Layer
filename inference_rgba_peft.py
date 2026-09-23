"""Compare PEFT RGBA checkpoints using identical prompts, seeds and sampling.

Only new PEFT checkpoints are accepted; inference_rgba.py remains unchanged.
"""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_from_local_dir, set_attention_backend
from zimage.pipeline import generate
from zimage.transparency import LatentTransparencyVAE
from zimage.peft_rgba import load_adapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--checkpoints", "--checkpoint", nargs="+", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=1.0,
                   help="Native pipeline convention; <=1 disables CFG. Keep identical across comparisons.")
    p.add_argument("--cfg-truncation", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--attention-backend", default="native")
    p.add_argument("--save-base-rgb", action="store_true",
                   help="Also decode the same generated latent with the frozen RGB VAE.")
    args = p.parse_args()
    if min(args.height, args.width, args.steps) < 1 or not 0 <= args.cfg_truncation <= 1:
        p.error("Invalid dimensions, steps or CFG truncation")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Seeds must be unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Use an empty output directory to avoid overwriting a comparison")
    required = ("COMPLETE", "adapter_config.json", "adapter_model.safetensors", "rgba_peft.json",
                "transparency_config.json", "transparency.safetensors")
    for checkpoint in args.checkpoints:
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete PEFT RGBA checkpoint {checkpoint}: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generation.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    set_attention_backend(args.attention_backend)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    for index, checkpoint in enumerate(args.checkpoints):
        # Reload the base each time; no adapter state leaks between checkpoints.
        components = load_from_local_dir(args.model_dir, device=args.device, dtype=dtype)
        codec = LatentTransparencyVAE.from_pretrained(components["vae"], checkpoint).requires_grad_(False).eval()
        components["transformer"], metadata = load_adapter(components["transformer"], checkpoint)
        if codec.transparency_config.latent_channels != components["transformer"].in_channels:
            raise ValueError("VAE and DiT latent channels differ")
        components["vae"] = codec
        prefix = f"{index:02d}-{checkpoint.name}"
        (args.output_dir / f"{prefix}-checkpoint.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with torch.no_grad():
            for seed in args.seeds:
                generator = torch.Generator(device=args.device).manual_seed(seed)
                latents = generate(**components, prompt=args.prompt, negative_prompt=args.negative_prompt,
                                   height=args.height, width=args.width, num_inference_steps=args.steps,
                                   guidance_scale=args.guidance_scale, cfg_truncation=args.cfg_truncation,
                                   generator=generator, output_type="latent")
                raw = codec.from_diffusion(latents.to(codec.dtype))
                decoded = codec.decode(raw, return_dict=False)[0]
                rgb = ((decoded[:, :3] + 1) / 2).clamp(0, 1)
                alpha = decoded[:, 3:4].sigmoid()
                rgba = torch.cat((rgb, alpha), dim=1)
                array = (rgba[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
                stem = f"{prefix}-seed-{seed}"
                image = Image.fromarray(array)
                image.save(args.output_dir / f"{stem}.png")
                image.getchannel("A").save(args.output_dir / f"{stem}-alpha.png")
                for name, color in (("black", (0, 0, 0, 255)), ("white", (255, 255, 255, 255))):
                    background = Image.new("RGBA", image.size, color)
                    background.alpha_composite(image)
                    background.convert("RGB").save(args.output_dir / f"{stem}-{name}.png")
                if args.save_base_rgb:
                    base = codec.base_vae.decode(raw, return_dict=False)[0][:, :3]
                    base = ((base[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
                    Image.fromarray((base * 255).round().astype("uint8")).save(args.output_dir / f"{stem}-base-rgb.png")
                    del base
                del latents, raw, decoded, rgb, alpha, rgba
                print(f"Saved {stem}", flush=True)
        del components, codec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
