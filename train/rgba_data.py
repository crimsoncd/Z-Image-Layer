"""RGBA PNG/TXT pairs with aspect buckets and alpha-aware resizing."""

import math
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler


def parse_buckets(specification, height, width):
    if specification:
        buckets = [tuple(int(value) for value in item.lower().split("x")) for item in specification.split(",")]
        if any(len(size) != 2 for size in buckets):
            raise ValueError("Buckets must be comma-separated WIDTHxHEIGHT pairs.")
        return buckets
    area = height * width
    return sorted({(max(16, round(math.sqrt(area * ratio) / 16) * 16),
                    max(16, round(math.sqrt(area / ratio) / 16) * 16))
                   for ratio in (0.5, 2 / 3, 0.8, 1.0, 1.25, 1.5, 2.0)} | {(width, height)})


class RGBATextDataset(Dataset):
    """Preserve the full image composition, including transparent margins."""

    def __init__(self, root, buckets):
        self.buckets = buckets
        self.samples = []
        self.bucket_indices = [[] for _ in buckets]
        for path in sorted(Path(root).rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".png" or not path.with_suffix(".txt").is_file():
                continue
            with Image.open(path) as image:
                if "A" not in image.getbands() and "transparency" not in image.info:
                    raise ValueError(f"Training image has no alpha channel: {path}")
                ratio = image.width / image.height
            bucket = min(range(len(buckets)), key=lambda i: abs(math.log(ratio / (buckets[i][0] / buckets[i][1]))))
            self.bucket_indices[bucket].append(len(self.samples))
            self.samples.append((path, bucket))
        if not self.samples:
            raise ValueError(f"No RGBA PNG/TXT pairs found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, bucket = self.samples[index]
        width, height = self.buckets[bucket]
        with Image.open(path) as source:
            image = source.convert("RGBA")
            scale = min(width / image.width, height / image.height)
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            # Resize premultiplied colors to avoid hidden RGB bleeding into edges.
            image = image.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        rgba = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        rgba = rgba.reshape(height, width, 4).permute(2, 0, 1).float() / 255
        return {"rgb": rgba[:3], "alpha": rgba[3:4],
                "caption": path.with_suffix(".txt").read_text(encoding="utf-8").strip()}


class AspectRatioBatchSampler(Sampler):
    def __init__(self, bucket_indices, batch_size):
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.drop_last = False

    def __iter__(self):
        batches = []
        for bucket in self.bucket_indices:
            indices = list(bucket)
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                # Complete a short batch inside its own bucket; distributed
                # padding must never mix images with different spatial shapes.
                if len(batch) < self.batch_size:
                    batch += random.choices(indices, k=self.batch_size - len(batch))
                batches.append(batch)
        random.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum(math.ceil(len(bucket) / self.batch_size) for bucket in self.bucket_indices)


@torch.no_grad()
def encode_captions(captions, tokenizer, text_encoder, device, max_length, dropout):
    formatted = [tokenizer.apply_chat_template(
        [{"role": "user", "content": "" if random.random() < dropout else caption}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for caption in captions]
    tokens = tokenizer(formatted, padding="max_length", truncation=True,
                       max_length=max_length, return_tensors="pt")
    masks = tokens.attention_mask.to(device).bool()
    hidden = text_encoder(input_ids=tokens.input_ids.to(device), attention_mask=masks,
                          output_hidden_states=True).hidden_states[-2]
    return [features[mask].detach() for features, mask in zip(hidden, masks)]
