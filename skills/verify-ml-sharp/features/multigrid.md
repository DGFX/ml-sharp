# Multi-grid finetune

Take the stage 3 run 2 checkpoint (`runs/stage3-b29749e/checkpoints/final.pt`, grid 192) and keep training at Gaussian grid 384 or 768. The convolutional decoder weights load onto a predictor rebuilt at the new grid. The base splat scale shrinks with the grid spacing (1/2 at 384, 1/4 at 768) so the warm start stays near the 192 quality.

## Sub-features

- Grid override on `stage2.load_checkpoint`. A checkpoint saved at one grid loads at another. Monodepth weights stay the Apple checkpoint, except head.4 when the checkpoint holds it. Decoder conv weights match.
- Base-scale prior. Measured median scale at the initializer's image width tracks `1 / grid`.
- Gaussian count. Grid 384, two layers: 294,912 Gaussians. Grid 768: 1,179,648.
- `--grad-checkpointing`. The multires conv decoder checkpoints. Decoder gradients stay non-zero.
- `--backbone-bf16`. Frozen Depth Pro runs in bfloat16 under `no_grad`. Decoder, covariances, gsplat and losses stay FP32. The backbone has no gradient.
- Budget. Default timeout and `--max-minutes` scale by `(grid / 192) ** 2`. GPU is `A100-40GB` below 768 and `H100` at 768.

CPU, before spending GPU time:

```
PYTHONPATH=src python -m sharp_train.multigrid
```

## How to get to it (user POV)

```
PYTHONPATH=src python -m sharp_train.multigrid
modal run modal_app.py --task stage3 --mode check --run-name stage3-grid384-check-<sha> --grid 384 --init-checkpoint runs/stage3-b29749e/checkpoints/final.pt --grad-checkpointing 1 --max-usd 6.69
modal run --detach modal_app.py --task stage3 --mode train --run-name stage3-grid384-<sha> --grid 384 --grad-checkpointing 1 --lr 1.6e-4 --warmup 300 --max-usd 17.84
modal run --detach modal_app.py --task stage3 --mode train --run-name stage3-grid768-<sha> --grid 768 --grad-checkpointing 1 --backbone-bf16 1 --lr 1.6e-4 --warmup 300 --timeout-minutes 840 --max-usd 58.8
```

The recipe is `--steps 8000`, `--lr 1.6e-4` and `--warmup 300` (passed explicitly, a new stage 3 run refuses without them), `--depth-correction 0` (the inference depth path, as in [stage 3](stage3.md)), paper loss weights and novel-view color inside the visibility mask. Run 2 held the depth alignment instead, and that mode no longer exists. Omit `--max-minutes` and `--timeout-minutes` to take the grid scale (x4 at 384, x16 at 768). Omit `--gpu` to take `A100-40GB` at 384 and `H100` at 768.

## Driving it with modal

Same `--task stage3` surface as [stage 3](stage3.md). Extra flags: `--grid`, `--grad-checkpointing`, `--backbone-bf16`. `--init-checkpoint` already defaults to the run 2 file.

## Evidence

- CPU JSON from `python -m sharp_train.multigrid`: `"passed": true`, `gaussians` 294912, `layers` 2, `scale_prior_ratio` 0.5, feature and head grad norms above 0 in both checkpointing modes, `backbone_bf16.block_grad` 0, `backbone_fp32.block_grad` above 0.
- On a GPU run, `train_meta.json` records `source_grid`, `scale_prior_ratio`, `grad_checkpointing`, `backbone_bf16`, `input_resolution` and `gpu`.

## Gotchas

- The count check forwards the real predictor at 1536 on the meta device. A full Depth Pro forward at 1536 does not fit a small CPU machine.
- Decoder gradients are measured on the real feature model and prediction head at a tiny feature pyramid. The frozen Depth Pro encoder rejects images below its 1536 working resolution, and one ViT-L/16 block is the backbone probe (legal input 384).
- The prediction head in Apple's checkpoint is not zero, which is why the feature tower gets a gradient. A freshly reset head does not.
- Eval reads the grid stored in the checkpoint. Pass `--grid` on an eval run anyway: it selects the GPU and the timeout (x4 at 384, x16 at 768).
- Checkpoints exclude the frozen monodepth weights but keep the trainable head.4, and inherit Apple's research-only model license.
