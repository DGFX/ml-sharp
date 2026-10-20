# Stage 3 losses

Continue from the stage 2 checkpoint. Add the paper's alpha BCE and perceptual loss on the novel view inside the frustum, the regularizers, and the depth-correction scale losses. Eval adds the hole fraction and LPIPS on the occluded region, for Apple at 768, Apple squeezed to 192, and ours.

## Sub-features

- `stage3 --mode check`: 30 training steps from the stage 2 final checkpoint with every stage 3 loss turned on. Prints steady-state seconds per step and peak GPU memory. Writes `stage3_check.json` with `"passed"`.
- `stage3 --mode train`: same init, Adam with warmup and cosine decay, novel-view color, alpha, and perceptual inside the frustum, novel-view disparity inside the visibility mask, regularizers every step. Wall-clock cap `--max-minutes`. Held-out validation includes the new terms. Checkpoints every `--checkpoint-every` steps and at the end.
- `stage3 --mode eval`: the stage 2 eval harness. `hole_fraction` is the share of novel-view pixels with rendered alpha below 0.5. `lpips_occluded` is AlexNet LPIPS after the render is replaced by the ground truth outside the occluded region. Side-by-side videos stay Apple on the left and ours on the right.

CPU, before spending GPU time:

```
PYTHONPATH=src python -m sharp_train.stage3
```

## How to get to it (user POV)

```
PYTHONPATH=src python -m sharp_train.stage3
modal run modal_app.py --task stage3 --mode check --run-name stage3b-check-<sha> --init stage2
modal run --detach modal_app.py --task stage3 --mode train --run-name stage3b-<sha> --init stage2 --steps 4000 --stage3-warmup 1000 --max-minutes 35 --timeout-minutes 45 --checkpoint-every 2000 --val-every 500
modal run modal_app.py --task stage3 --mode eval --run-name stage3b-<sha> --checkpoints final --timeout-minutes 20
```

`A100-80GB` when 40 GB is tight:

```
modal run --detach modal_app.py --task stage3 --mode train --run-name stage3-<sha> --gpu A100-80GB --init stage2 --steps 8000 --stage3-warmup 1000 --max-minutes 90 --timeout-minutes 120
STAGE3_GPU=A100-80GB modal run modal_app.py --task stage3 --mode check --run-name stage3-check-<sha>
```

## Driving it with modal

`--mode` is `check`, `train` or `eval`. `--init stage2` (default) loads `runs/stage2-29fefbe/checkpoints/final.pt`. `--init stage3b` loads `runs/stage3-b29749e/checkpoints/final.pt`. A non-empty `--init-checkpoint` overrides `--init`. `--stage3-warmup` defaults to 1000. `--stage3-warmup 10000` is the paper ramp; on 4000 steps that stays on the ramp and finishes at 6.4e-5. The default 1000 reaches 1.6e-4 at step 999 and is on the cosine by the end of a 4000-step run (about 1.60e-5). `--percep-weight` defaults to 3. `--warmup` is the stage 2 schedule and stage 3 does not read it. `--depth-correction 0` turns the scale map off. `--gpu` or `STAGE3_GPU` overrides `A100-40GB`. `--timeout-minutes` overrides the function timeout (45 min check, 120 min train, 60 min eval). Train also takes `--grid`, `--steps`, `--lr`, `--max-minutes`, `--checkpoint-every`, `--val-every`, `--seed`, `--train-scenes`. Eval takes `--checkpoints`. The budget-capped probe is 4000 steps from stage 2: about $1.00 of training at $0.25/1000 steps, plus an eval timeout of 20 minutes (about $0.70 on an A100-40GB). The whole run stays under about $1.80 even if eval hits that timeout.

## Evidence

- CPU: one JSON object, `"passed": true`.
- `stage3_check.json`: `seconds_per_step`, `peak_memory_gb`, `last` (every loss term), `"passed"`.
- `history.json`, `val_history.json`, `train_curve.png`, `val_curve.png`, `train_meta.json`, `checkpoints/*.pt`.
- `eval/<checkpoint>/report.json` with `mean.<setting>.<model>.hole_fraction` and `lpips_occluded` for `apple_768`, `apple_decoder_<grid>` and ours. Same comparison images and videos as stage 2.

## Gotchas

- Novel-view color, alpha, and perceptual use the frustum mask. Disocclusions inside the frustum are supervised; pixels outside it are not. Novel-view disparity stays on the visibility mask. A drop in `hole_fraction` and `lpips_occluded` is the signal that disocclusions are filling in; full-image LPIPS can lag behind. Eval still scores those on the visibility mask.
- Depth correction is on by default and is already what stage 2 trained. Turning it off is an ablation, not the starting point. Stage 3 freezes that module (`hold_depth_alignment`). While the aligned depth and the scale map do not require grad, second-layer TV, scale MAE, and multiscale scale TV are 0 and are not computed. They still run when those tensors require grad. The floater penalty still multiplies both layers' opacities.
- `history.json` `grad_norm` is the norm before `clip_grad_norm_` at 1.0. `train_meta.json` gets `peak_memory_gb` during the loop.
- Stage 3 warmup defaults to 1,000 steps. `--stage3-warmup 10000` is the paper ramp. The held run used 300, so by step 500 the learning rate was already at the peak while perceptual loss was most of the total. That enlarged splats past the unscaled variance cap of 100 (splat loss 0.48 is the upper hinge; the lower hinge cannot exceed 0.1) and raised val color L1. The cap now scales by `(768/grid)^2`. Start the next run from `--init stage2`.
- The perceptual net is an ImageNet ResNet-50, baked into the Modal image. The paper's inpainting ResNet-50 is not used. See `docs/training-spec.md`.
- Checkpoints still exclude monodepth weights and inherit Apple's research-only model license. Renders are Hypersim: credit Hypersim (CC BY-SA 3.0).
