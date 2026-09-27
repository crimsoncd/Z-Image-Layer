#!/usr/bin/env bash
set -euo pipefail

MODEL=/remote-home/Zhangkaile/models/Z-Image
CODEC=/remote-home/Zhangkaile/dev/Z-Image-Layer/outputs/magick100k/stage1/checkpoint-100000
DATA_DIRS=(
  /remote-home/Zhangkaile/datasets/MAGICK/100K
  /remote-home/Zhangkaile/datasets/MAGICK/extra
)
RUN=/remote-home/Zhangkaile/dev/Z-Image-Layer/outputs/magick100k/peft-v2-r64
LOG=/remote-home/Zhangkaile/dev/Z-Image-Layer/logs/peft-v2-r64.log

# To continue a run, add for example:
#   --resume-from-checkpoint "$RUN/checkpoint-10000"
nohup python train/train_rgba_peft_v2.py \
  --model-dir "$MODEL" --codec-dir "$CODEC" \
  --data-dir "${DATA_DIRS[@]}" \
  --output-dir "$RUN" \
  --groups attention,mlp --scopes layers,noise_refiner \
  --rank 64 --lora-alpha 64 \
  --buckets 512x512 --batch-size 4 --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 --mixed-precision bf16 \
  --max-steps 100000 --save-every 10000 --save-steps 1 --log-every 1000 \
  --num-workers 4 > "$LOG" 2>&1 &
