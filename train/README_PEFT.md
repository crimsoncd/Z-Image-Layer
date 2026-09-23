# Z-Image RGBA training with PEFT

Independent stage-2 entry points. Existing training/inference files are unchanged.
This uses PEFT's [low-level API](https://huggingface.co/docs/peft/main/en/developer_guides/low_level_api):
`inject_adapter_in_model`, `get_peft_model_state_dict`, and `set_peft_model_state_dict`.
No model training, inference or runtime tests were executed during implementation.

## Environment and inputs

Run from the repository root on the Linux training server. Prefer a separate environment
while existing experiments are running; dependency installation may upgrade Transformers.
Install the existing project dependencies, then the PEFT dependency:

```bash
python -m pip install -e .
python -m pip install -r train/requirements-peft.txt

MODEL=/remote-home/Zhangkaile/models/Z-Image
CODEC=/remote-home/Zhangkaile/dev/Z-Image-Layer/outputs/magick10k/stage1/checkpoint-6000
DATA=/remote-home/Zhangkaile/datasets/MAGICK/10K
RUN=outputs/magick10k/peft-attn-mlp
```

Use the same Z-Image Base weights and RGBA codec throughout an experiment.
`CODEC` must contain `transparency_config.json` and `transparency.safetensors`.
The dataset uses paired RGBA PNG / UTF-8 TXT files, recursively, as in the existing trainer.
Output directories must be empty/new; the scripts refuse to overwrite experiments.

## Train

```bash
python train/train_rgba_peft.py \
  --model-dir "$MODEL" --codec-dir "$CODEC" --data-dir "$DATA" \
  --output-dir "$RUN" \
  --groups attention,mlp --scopes layers,noise_refiner \
  --rank 32 --lora-alpha 32 \
  --buckets 512x512 --batch-size 4 --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 --mixed-precision bf16 \
  --max-steps 10000 --save-every 1000 --save-steps 1,100,500 \
  --num-workers 4
```

The codec/text encoder/base DiT weights are frozen; only injected PEFT parameters train.
Flow objective, timestep sampling and latent scaling follow the existing stage-2 implementation.
Changing PEFT targets alone does not guarantee better RGBA quality. No alpha reconstruction
loss or full-parameter tuning has been silently added.

`max-steps` counts optimizer updates, not microbatches. Effective batch is
`batch-size * gradient-accumulation-steps * number_of_processes`.
BF16 applies to the frozen DiT; trainable adapter parameters are stored in FP32.
The script supports ordinary Accelerate/DDP launching, but does not configure FSDP/ZeRO.
Arbitrary target selections may contain unused modules; DDP unused-parameter detection is enabled.
There is no gradient checkpointing/offload in this new entry point.

## Choose injection locations

`--groups` is a comma-separated union:

| Group | Linear modules |
|---|---|
| `attention` | `attention.to_q`, `to_k`, `to_v`, `to_out.0` |
| `mlp` | `feed_forward.w1`, `w2`, `w3` |
| `modulation` | block `adaLN_modulation` |
| `input` | `all_x_embedder.2-1` |
| `output` | `all_final_layer.2-1.linear` (not final modulation) |
| `time` | linear layers within `t_embedder` |
| `text` | linear layers within `cap_embedder` |

`--scopes layers,noise_refiner,context_refiner` selects scopes for attention/MLP/block
modulation only. Input/output/time/text groups are global and do not use scopes/blocks.
`--blocks 0-3,20,25-29` restricts **main `layers` only**, with zero-based inclusive indices;
it does not restrict refiner indices. Invalid main indices are rejected.

Examples:

```bash
# Reproduce the old injection coverage (all attention, including text refiner):
--groups attention --scopes layers,noise_refiner,context_refiner

# Broader image/main attention + MLP, excluding the text refiner:
--groups attention,mlp --scopes layers,noise_refiner

# Also add LoRA to input and final output projections (NOT full weight tuning):
--groups attention,mlp,input,output --scopes layers,noise_refiner

# Only main blocks 0 through 3:
--groups attention,mlp --scopes layers --blocks 0-3

# Custom exact full-match regex replaces groups/scopes/blocks:
--target-regex 'layers\.(0|1|2|3)\.feed_forward\.w[123]'

# Exclude selected key projections by regex search:
--exclude-regex '\.to_k$'
```

Inspect actual module names before launching (loads model weights, does not save/train):

```bash
python train/train_rgba_peft.py \
  --model-dir "$MODEL" --codec-dir "$CODEC" --output-dir "$RUN" \
  --groups attention,mlp,input,output --list-targets
```

Exact target names are printed and saved in every checkpoint's `rgba_peft.json`.
Custom regexes may select unused alternative patch heads; select the active `2-1` heads.
Use `--use-dora` or `--use-rslora` for PEFT variants. These change the adaptation method;
do not mix their results with standard LoRA without recording the configuration.

## Step 0 and step 1

Every run saves `checkpoint-0` before training. Standard LoRA uses random A / zero B,
so a fresh step-0 adapter initially leaves the DiT function unchanged (up to numeric effects).
The **trained RGBA decoder is still active**, so step-0 RGBA is not expected to look like
the original RGB model's output or have correct transparency.

Export only step 0, without loading any dataset:

```bash
python train/train_rgba_peft.py \
  --model-dir "$MODEL" --codec-dir "$CODEC" \
  --output-dir outputs/magick10k/peft-zero \
  --groups attention,mlp --max-steps 0
```

For a one-update run, pass `--max-steps 1` and `--data-dir "$DATA"` instead.
This saves both checkpoint-0 and checkpoint-1. A long run also saves step 1 by default.
Compare checkpoints from the **same run** to hold initialization and codec fixed.

## Checkpoints and warm starts

```text
checkpoint-1000/
  adapter_config.json          # PEFT LoraConfig
  adapter_model.safetensors    # adapter weights only
  transparency_config.json    # frozen RGBA codec architecture
  transparency.safetensors    # frozen RGBA codec weights
  rgba_peft.json              # step, targets, arguments, PEFT version, source paths
  COMPLETE                    # written after successful saving
```

The original RGB VAE and DiT are not duplicated; inference still needs `--model-dir`.
Checkpoints are not compatible with the old `inference_rgba.py` or `--init-lora` format.
`metrics.jsonl` contains interval sample-weighted mean loss across processes, latest local
microbatch loss, rank-0 gradient norm before clipping, and actual learning rate.
It is not a fixed validation-set metric. Parse JSON to preserve scientific-notation learning rates.

`--init-adapter PATH` is an optional **weights-only warm start**. It loads the saved adapter
configuration (target/rank flags are ignored), while `--codec-dir` must select the matching
codec, preferably the same PATH. Optimizer/RNG/data position are not restored and local step
count restarts at zero. Such a checkpoint-0 is marked as a warm start, not untrained.
The scripts record base paths but cannot verify identical remote weights; use the exact same base.

## Compare generations on the server

```bash
python inference_rgba_peft.py \
  --model-dir "$MODEL" \
  --checkpoints "$RUN/checkpoint-0" "$RUN/checkpoint-1" "$RUN/checkpoint-1000" \
  --prompt "A whole red apple, isolated on a transparent background." \
  --seeds 42 123 2026 --height 512 --width 512 --steps 30 \
  --guidance-scale 1.0 --save-base-rgb \
  --output-dir outputs/magick10k/peft-comparison
```

Each checkpoint gets the same initial noise per seed. Outputs include RGBA, alpha grayscale,
black/white composites, optional frozen-VAE RGB reconstruction of the **same generated latent**,
and generation/checkpoint metadata. The latter RGB image is not a separate unadapted DiT sample.
The base is reloaded for each checkpoint to prevent adapter contamination.

The command deliberately uses the existing native pipeline convention: `guidance-scale <= 1`
disables CFG; larger values use `positive + scale * (positive - negative)`. No CFG correction
is made to the original pipeline. For a Base model CFG experiment, use a separate output directory
with a chosen larger value and hold it fixed across all steps. Do not compare checkpoints with
different guidance, prompts, seeds or codec weights and attribute differences solely to training.

For caption ablations, `--caption-suffix 'on a transparent background.'` adds the phrase in
memory without editing TXT files. Caption dropout still drops the entire resulting caption.
