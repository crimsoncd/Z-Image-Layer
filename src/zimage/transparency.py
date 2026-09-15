"""LayerDiffuse-inspired latent transparency for the native Z-Image VAE.

The frozen RGB VAE retains its latent channel count. A bounded RGBA encoder
adds transparency to that latent, and a separate decoder reads both the latent
and the frozen VAE's RGB reconstruction. This is an adaptation, not a binary
compatible implementation of LayerDiffuse's checkpoints.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F

from .autoencoder import AutoencoderKLOutput


@dataclass
class TransparencyConfig:
    latent_channels: int
    scale_factor: int
    hidden_channels: int = 32
    offset_scale: float = 0.1
    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError(f"Unsupported transparency format: {self.version}")
        if self.scale_factor < 1 or self.scale_factor & (self.scale_factor - 1):
            raise ValueError("VAE scale_factor must be a positive power of two.")
        if self.latent_channels < 1 or self.hidden_channels < 8 or self.hidden_channels % 8:
            raise ValueError("latent_channels must be positive; hidden_channels must be a multiple of 8.")
        if self.offset_scale <= 0:
            raise ValueError("offset_scale must be positive.")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class TransparencyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_channels
        blocks = [nn.Conv2d(4, width, 3, padding=1), ResidualBlock(width)]
        for _ in range(config.scale_factor.bit_length() - 1):
            next_width = min(width * 2, config.hidden_channels * 4)
            blocks.extend([nn.Conv2d(width, next_width, 4, stride=2, padding=1), ResidualBlock(next_width)])
            width = next_width
        self.features = nn.Sequential(*blocks)
        self.out = nn.Conv2d(width, config.latent_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.offset_scale = config.offset_scale

    def forward(self, rgba):
        return self.offset_scale * self.out(self.features(rgba)).tanh()


class TransparencyDecoder(nn.Module):
    """RGB U-Net with latent conditioning at its bottleneck and fine RGB skips."""

    def __init__(self, config):
        super().__init__()
        widths = [min(config.hidden_channels * 2**i, config.hidden_channels * 4)
                  for i in range(config.scale_factor.bit_length())]
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1), ResidualBlock(widths[0]))
        self.down = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b, 4, stride=2, padding=1), ResidualBlock(b))
            for a, b in zip(widths, widths[1:])
        ])
        self.latent = nn.Conv2d(config.latent_channels, widths[-1], 3, padding=1)
        self.mid = ResidualBlock(widths[-1])
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a + b, b, 3, padding=1), ResidualBlock(b))
            for a, b in zip(reversed(widths[1:]), reversed(widths[:-1]))
        ])
        self.out = nn.Conv2d(widths[0], 4, 3, padding=1)

    def forward(self, rgb, latent):
        x = self.stem(rgb)
        skips = [x]
        for block in self.down:
            x = block(x)
            skips.append(x)
        x = self.mid(x + self.latent(latent))
        for block, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        output = self.out(x)
        # Native pipeline contract: RGB in [-1, 1], alpha as logits.
        return torch.cat((output[:, :3].tanh(), output[:, 3:4]), dim=1)


class LatentTransparencyVAE(nn.Module):
    """Wrap a pretrained RGB VAE without changing the DiT latent interface.

    encode_rgba consumes straight RGB and alpha in [0, 1] and returns raw VAE
    latents. to_diffusion/from_diffusion are the only scaling boundaries.
    decode consumes raw VAE latents, matching pipeline.generate's convention.
    """

    def __init__(self, base_vae, config=None):
        super().__init__()
        self.base_vae = base_vae.requires_grad_(False).eval()
        self.config = base_vae.config
        channels = self.config.latent_channels
        factor = 2 ** (len(self.config.block_out_channels) - 1)
        self.transparency_config = config or TransparencyConfig(channels, factor)
        if (channels, factor) != (self.transparency_config.latent_channels, self.transparency_config.scale_factor):
            raise ValueError("Transparency checkpoint does not match this VAE's latent shape.")
        self.transparency_encoder = TransparencyEncoder(self.transparency_config)
        self.transparency_decoder = TransparencyDecoder(self.transparency_config)
        self.to(device=next(base_vae.parameters()).device, dtype=torch.float32)

    @property
    def dtype(self):
        return next(self.base_vae.parameters()).dtype

    def train(self, mode=True):
        super().train(mode)
        self.base_vae.eval()
        return self

    def to_diffusion(self, latent):
        return (latent - (self.config.shift_factor or 0.0)) * self.config.scaling_factor

    def from_diffusion(self, latent):
        return latent / self.config.scaling_factor + (self.config.shift_factor or 0.0)

    def encode_rgba(self, rgb, alpha, return_aux=False):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or alpha.shape != (rgb.shape[0], 1, *rgb.shape[-2:]):
            raise ValueError("Expected RGB [B,3,H,W] and alpha [B,1,H,W].")
        if any(size % self.transparency_config.scale_factor for size in rgb.shape[-2:]):
            raise ValueError("Image dimensions must be divisible by the VAE scale factor.")
        # Premultiplication removes arbitrary hidden colors. Alpha remains an
        # explicit encoder input, distinguishing black objects from empty pixels.
        premultiplied = rgb * alpha
        base_input = premultiplied * 2 - 1
        with torch.no_grad():
            moments = self.base_vae.encoder(base_input)
            if self.base_vae.quant_conv is not None:
                moments = self.base_vae.quant_conv(moments)
            base_latent = moments.chunk(2, dim=1)[0]
        offset = self.transparency_encoder(torch.cat((base_input, alpha * 2 - 1), dim=1))
        latent = base_latent + offset
        return (latent, base_latent, offset) if return_aux else latent

    def decode(self, latent, return_dict=True):
        # Frozen parameters still propagate reconstruction gradients to offset.
        rgb = self.base_vae.decode(latent, return_dict=False)[0][:, :3]
        decoded = self.transparency_decoder(rgb, latent)
        return AutoencoderKLOutput(sample=decoded) if return_dict else (decoded,)

    def forward(self, rgb, alpha):
        latent, base_latent, offset = self.encode_rgba(rgb, alpha, return_aux=True)
        rgb_reconstruction = self.base_vae.decode(latent, return_dict=False)[0][:, :3]
        decoded = self.transparency_decoder(rgb_reconstruction, latent)
        with torch.no_grad():
            original = self.base_vae.decode(base_latent, return_dict=False)[0][:, :3]
        return {"decoded": decoded, "offset": offset,
                "identity_prediction": rgb_reconstruction, "identity_target": original}

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.transparency_config)
        payload["vae_scaling_factor"] = self.config.scaling_factor
        payload["vae_shift_factor"] = self.config.shift_factor or 0.0
        (directory / "transparency_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        weights = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()
                   if not name.startswith("base_vae.")}
        save_file(weights, str(directory / "transparency.safetensors"))

    @classmethod
    def from_pretrained(cls, base_vae, directory):
        directory = Path(directory)
        payload = json.loads((directory / "transparency_config.json").read_text(encoding="utf-8"))
        if (payload.pop("vae_scaling_factor") != base_vae.config.scaling_factor
                or payload.pop("vae_shift_factor") != (base_vae.config.shift_factor or 0.0)):
            raise ValueError("Transparency checkpoint VAE scaling/shift does not match the base model.")
        model = cls(base_vae, TransparencyConfig(**payload))
        state = load_file(str(directory / "transparency.safetensors"))
        expected = {key for key in model.state_dict() if not key.startswith("base_vae.")}
        if set(state) != expected:
            raise ValueError("Incomplete or incompatible transparency checkpoint.")
        model.load_state_dict(state, strict=False)
        return model


def transparency_losses(outputs, rgb, alpha):
    """Soft-alpha supervision, visible color, compositing and latent identity."""
    decoded = outputs["decoded"].float()
    predicted_rgb = (decoded[:, :3] + 1) / 2
    logits = decoded[:, 3:4]
    predicted_alpha = logits.sigmoid()
    rgb, alpha = rgb.float(), alpha.float()
    background = torch.rand_like(rgb)
    predicted_composite = predicted_rgb * predicted_alpha + background * (1 - predicted_alpha)
    target_composite = rgb * alpha + background * (1 - alpha)
    edge = sum(F.l1_loss(torch.diff(predicted_alpha, dim=dim), torch.diff(alpha, dim=dim))
               for dim in (-1, -2))
    return {
        "alpha": F.binary_cross_entropy_with_logits(logits, alpha),
        "edge": edge,
        "rgb": ((predicted_rgb - rgb).abs() * alpha).sum() / (3 * alpha.sum()).clamp_min(1),
        "composite": F.l1_loss(predicted_composite, target_composite),
        "identity": F.mse_loss(outputs["identity_prediction"].float(), outputs["identity_target"].float()),
        "offset": outputs["offset"].float().square().mean(),
    }
