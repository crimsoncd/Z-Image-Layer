#!/usr/bin/env bash
set -euo pipefail

MODEL=/remote-home/Zhangkaile/models/Z-Image
CODEC=/remote-home/Zhangkaile/dev/Z-Image-Layer/outputs/magick100k/stage1/checkpoint-100000
IMAGES=/remote-home/Zhangkaile/datasets/MAGICK/VAEExam
OUTPUT=/remote-home/Zhangkaile/dev/Z-Image-Layer/test/reconstruction/VAEExam-100000

for i in 001 002 003 004 005 006 007 008 009; do
    python test/test_rgba_vae.py \
        --model-dir "$MODEL" \
        --checkpoint "$CODEC" \
        --image "$IMAGES/$i.png" \
        --output-dir "$OUTPUT/$i/"
done
