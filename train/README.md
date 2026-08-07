# Alpha-channel training

The trainer expects one directory containing paired files:

```text
dataset/
  0001.png
  0001.txt
  0002.png
  0002.txt
```

PNGs are loaded as RGBA. RGB is encoded by the pretrained VAE; alpha supervises
the new decoder head. The default objective combines transformer flow matching,
balanced alpha BCE, soft Dice, alpha-edge, random-background compositing, and a
small foreground RGB reconstruction loss.

Run from the repository root:

```powershell
D:\Miniconda\envs\Steel\python.exe train\train_alpha.py `
  --model-dir ckpts\Z-Image-Turbo `
  --data-dir D:\path\to\dataset `
  --output-dir outputs\alpha-training `
  --height 512 --width 512 `
  --batch-size 1 --gradient-accumulation-steps 4
```

Useful lower-memory variants:

```powershell
# Train only the alpha head; useful as the first stage.
--transformer-train-mode none --decoder-train-mode alpha

# Train the final four transformer blocks and the alpha head.
--transformer-train-mode last --transformer-last-layers 4
```

Each checkpoint stores only parameters selected for training:
`transformer_trainable.safetensors`, `vae_trainable.safetensors`, and
`training_state.json`. Pass the checkpoint directory to `--resume` to load those
weights. Optimizer state is intentionally not stored, so resuming restarts the
learning-rate schedule while retaining `global_step` for naming/logging.

## Test a checkpoint

The test script loads the original model first and then applies the trainable
weight files from a checkpoint:

```powershell
D:\Miniconda\envs\Steel\python.exe train\test_alpha.py `
  --model-dir ckpts\Z-Image-Turbo `
  --checkpoint outputs\alpha-training\checkpoint-1000 `
  --prompt "a ceramic teapot, isolated object, transparent background" `
  --output outputs\teapot.png `
  --height 512 --width 512 --seed 42
```

The saved PNG is RGBA. The script also prints the minimum, maximum, and mean
opacity so a fully opaque or collapsed alpha prediction is easy to notice.
