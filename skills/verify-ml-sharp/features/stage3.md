# Stage 3 losses

Continue from the stage 3 run 2 checkpoint, which was trained from stage 2. The losses are the paper's alpha BCE and perceptual loss on the full novel view, the regularizers, and the depth-correction scale losses. Eval adds the hole fraction and LPIPS on the occluded region, for Apple at 768, Apple squeezed to the training grid, and ours.

## Sub-features

- `stage3 --mode check`: 30 training steps from the stage 3 run 2 checkpoint with every stage 3 loss turned on. Prints steady-state seconds per step and peak GPU memory. Writes `stage3_check.json` with `"passed"`.
- `stage3 --mode train`: same init, Adam with warmup and cosine decay, novel-view color and disparity still inside the visibility mask, alpha and perceptual on the full novel view, regularizers every step. Wall-clock cap `--max-minutes`. Held-out validation includes the new terms. Checkpoints every `--checkpoint-every` steps and at the end.
- `stage3 --mode eval`: the stage 2 eval harness. `hole_fraction` is the share of novel-view pixels with rendered alpha below 0.5. `lpips_occluded` is AlexNet LPIPS after the render is replaced by the ground truth outside the occluded region. Side-by-side videos stay Apple on the left and ours on the right.

CPU, before spending GPU time:

```
PYTHONPATH=src python -m sharp_train.stage3
```

## How to get to it (user POV)

```
PYTHONPATH=src python -m sharp_train.stage3
modal run modal_app.py --task stage3 --mode check --run-name stage3-check-<sha>
modal run --detach modal_app.py --task stage3 --mode train --run-name stage3-<sha> --steps 8000 --max-minutes 90 --timeout-minutes 120
modal run modal_app.py --task stage3 --mode eval --run-name stage3-<sha> --checkpoints final
```

Grid 384 stays on `A100-40GB`. Grid 768 selects `H100` unless `--gpu` or `STAGE3_GPU` is set. Timeouts and the in-loop budget scale by `(grid / 192) ** 2` when `--timeout-minutes` and `--max-minutes` are omitted.

```
modal run --detach modal_app.py --task stage3 --mode train --run-name stage3-grid384-<sha> --grid 384 --grad-checkpointing 1
modal run --detach modal_app.py --task stage3 --mode train --run-name stage3-grid768-<sha> --grid 768 --grad-checkpointing 1 --backbone-bf16 1
```

## Driving it with modal

`--mode` is `check`, `train` or `eval`. `--init-checkpoint` defaults to `runs/stage3-b29749e/checkpoints/final.pt` on the volume (stage 3 run 2, grid 192). `--grid` rebuilds the predictor at 384 or 768 and loads the decoder weights; the base splat scale shrinks with the grid spacing. `--depth-correction 0` turns the scale map off. `--grad-checkpointing 1` checkpoints the Gaussian decoder. `--backbone-bf16 1` runs the frozen Depth Pro forward in bfloat16 with no gradient. `--gpu` or `STAGE3_GPU` overrides `A100-40GB` (or `H100` at grid 768). `--timeout-minutes` overrides the scaled function timeout (45, 120 and 60 minutes at grid 192 for check, train and eval). Train also takes `--steps` (8000), `--lr` (`1.6e-4`), `--warmup` (300), `--max-minutes`, `--checkpoint-every`, `--val-every`, `--seed`, `--train-scenes`. Eval takes `--checkpoints`.

## Evidence

- CPU: one JSON object, `"passed": true`.
- `stage3_check.json`: `seconds_per_step`, `peak_memory_gb`, `last` (every loss term), `"passed"`.
- `history.json`, `val_history.json`, `train_curve.png`, `val_curve.png`, `train_meta.json`, `checkpoints/*.pt`.
- `eval/<checkpoint>/report.json` with `mean.<setting>.<model>.hole_fraction` and `lpips_occluded` for `apple_768`, `apple_decoder_<grid>` and ours. Same comparison images and videos as stage 2.

## Gotchas

- The novel-view color loss is still masked. Only alpha and perceptual see the holes. A drop in `hole_fraction` and `lpips_occluded` is the signal that disocclusions are filling in; full-image LPIPS can lag behind.
- Depth correction is on by default and is already what stage 2 trained. Turning it off is an ablation, not the starting point. Stage 3 freezes that module (`hold_depth_alignment`) so the scale regularizer and the second-layer TV cannot push the stage 2 scale off the ground truth. The scale losses stay in the total at the paper weights.
- `history.json` `grad_norm` is the norm before `clip_grad_norm_` at 1.0. `train_meta.json` gets `peak_memory_gb` during the loop.
- The perceptual net is an ImageNet ResNet-50, baked into the Modal image. The paper's inpainting ResNet-50 is not used. See `docs/training-spec.md`.
- Checkpoints still exclude monodepth weights and inherit Apple's research-only model license. Renders are Hypersim: credit Hypersim (CC BY-SA 3.0).
