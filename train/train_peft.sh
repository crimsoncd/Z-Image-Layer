
MODEL=/remote-home/Zhangkaile/models/Z-Image
CODEC=/remote-home/Zhangkaile/dev/Z-Image-Layer/outputs/magick100k/stage1/checkpoint-100000
DATA=/remote-home/Zhangkaile/datasets/MAGICK/100K
RUN=/remote-home/Zhangkaile/dev/Z-Image-Layer/outputs/magick100k/peft-r64-long

LOG=/remote-home/Zhangkaile/dev/Z-Image-Layer/logs/0924-2113-peft-r64-long.log



nohup python train/train_rgba_peft.py \
  --model-dir "$MODEL" --codec-dir "$CODEC" --data-dir "$DATA" \
  --output-dir "$RUN" \
  --groups attention,mlp --scopes layers,noise_refiner \
  --rank 64 --lora-alpha 64 \
  --buckets 512x512 --batch-size 4 --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 --mixed-precision bf16 \
  --max-steps 100000 --save-every 10000 --save-steps 1 --log-every 1000 \
  --num-workers 4 > "$LOG" 2>&1 &
