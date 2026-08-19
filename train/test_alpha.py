"""Load an alpha-training checkpoint and generate an RGBA test image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import load_from_local_dir, set_attention_backend  # noqa: E402
from zimage import generate  # noqa: E402


DEFAULT_PROMPT  = "A striped orange and cream canopy market stall with two vendors wearing aprons, one in blue and one in yellow, standing behind a counter with baskets of produce and hanging yellow lanterns."
DEFAULT_MODEL_PATH = "/remote-home/Zhangkaile/models/Z-Image/"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_PATH, help="Original Z-Image model directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint-N directory.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=Path("train/alpha-test.png"))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention-backend", default="_native_flash")
    return parser.parse_args()


def apply_delta(module: torch.nn.Module, path: Path, prefix_to_remove: str = "") -> int:
    if not path.is_file():
        return 0

    state = load_file(str(path), device="cpu")
    if prefix_to_remove:
        state = {
            key[len(prefix_to_remove) :] if key.startswith(prefix_to_remove) else key: value
            for key, value in state.items()
        }

    module_keys = set(module.state_dict())
    unexpected = sorted(set(state) - module_keys)
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise ValueError(f"Checkpoint {path} has unexpected keys: {preview}")

    module.load_state_dict(state, strict=False)
    return len(state)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Pass --device cpu if intentional.")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    set_attention_backend(args.attention_backend)

    components = load_from_local_dir(
        args.model_dir,
        device=args.device,
        dtype=dtype,
        verbose=True,
    )
    transformer_count = apply_delta(
        components["transformer"],
        args.checkpoint / "transformer_trainable.safetensors",
    )
    vae_count = apply_delta(
        components["vae"],
        args.checkpoint / "vae_trainable.safetensors",
        prefix_to_remove="vae.",
    )
    if transformer_count == 0 and vae_count == 0:
        raise FileNotFoundError(f"No trainable-weight files found in {args.checkpoint}")

    components["transformer"].eval()
    components["vae"].eval()
    print(f"Loaded {transformer_count} transformer tensors and {vae_count} VAE tensors.")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    images = generate(
        **components,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pil",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = images[0]
    if image.mode != "RGBA":
        raise RuntimeError(f"Expected an RGBA result, but pipeline returned {image.mode}.")
    image.save(args.output)

    alpha = torch.frombuffer(bytearray(image.getchannel("A").tobytes()), dtype=torch.uint8).float().div(255.0)
    print(
        f"Saved {args.output} | alpha min={alpha.min().item():.4f} "
        f"max={alpha.max().item():.4f} mean={alpha.mean().item():.4f}"
    )


if __name__ == "__main__":
    main()
