# Stage 3 losses

Continue from the stage 3 run 2 checkpoint, which was trained from stage 2. The losses are the paper's alpha BCE and perceptual loss on the full novel view, the regularizers, and the depth-correction scale losses. Eval adds the hole fraction and LPIPS on the occluded region, for Apple at 768, Apple squeezed to the training grid, and ours.

## Sub-features

- `stage3 --mode check`: 30 training steps from the stage 3 run 2 checkpoint with every stage 3 loss turned on. Prints steady-state seconds per step, peak GPU memory, `grad_clip`, `grad_norm_max` and `clipped_fraction`. Writes `stage3_check.json` with `"passed"`.
- `stage3 --mode train`: same init, Adam with warmup and cosine decay, novel-view color and disparity still inside the visibility mask, alpha and perceptual on the full novel view, regularizers every step. Wall-clock cap `--max-minutes`. Held-out validation runs on the 8 `ai_011_001` pairs in `val_pairs.json` and includes the new terms. Checkpoints every `--checkpoint-every` steps and at the end.
- `stage3 --mode eval`: the stage 2 eval harness. `hole_fraction` is the share of novel-view pixels with rendered alpha below 0.5. `hole_fraction_frustum` is the same share inside the input frustum only. `lpips_occluded` is AlexNet LPIPS after the render is replaced by the ground truth outside the occluded region. Side-by-side videos stay Apple on the left and ours on the right.

CPU, before spending GPU time (see the CPU checks section of `SKILL.md` for the data):

```
PYTHONPATH=src python -m sharp_train.stage3 --data ~/data/hypersim-cpu
```

## How to get to it (user POV)

```
PYTHONPATH=src python -m sharp_train.stage3 --data ~/data/hypersim-cpu
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

`--mode` is `check`, `train` or `eval`. `--init-checkpoint` defaults to `runs/stage3-b29749e/checkpoints/final.pt` on the volume (stage 3 run 2, grid 192). `--grid` rebuilds the predictor at 384 or 768 and loads the decoder weights; the base splat scale shrinks with the grid spacing. `--depth-correction` defaults to 0, the inference depth path, so training and eval see the same depth. `--depth-correction 1` feeds ground truth to the alignment and trains it under the scale losses. `--grad-clip` is the global grad-norm clip (default 1000, 0 turns it off). `--grad-checkpointing 1` checkpoints the Gaussian decoder. `--backbone-bf16 1` runs the frozen Depth Pro forward in bfloat16 with no gradient. `--gpu` or `STAGE3_GPU` overrides `A100-40GB` (or `H100` at grid 768). `--timeout-minutes` overrides the scaled function timeout (45, 120 and 60 minutes at grid 192 for check, train and eval). Train also takes `--steps` (8000), `--lr` (`1.6e-4`), `--warmup` (300), `--max-minutes`, `--checkpoint-every`, `--val-every`, `--seed`, `--train-scenes`, `--kill-val-ratio` (0 never stops; `R` stops at the first validation total above `R` times step 0 and saves `final.pt`). Steps with a pre-clip grad norm above 10,000 write per-term grad norms to `spikes.json`. Eval takes `--checkpoints`. `python scripts/compare_eval.py --base A/report.json --new B/report.json` is the pass gate against a previous run; it allows `--noise-k` (default 1) paired standard errors of noise per gated metric.

## Evidence

- CPU: one JSON object, `"passed": true`.
- `stage3_check.json`: `seconds_per_step`, `peak_memory_gb`, `last` (every loss term), `"passed"`.
- `history.json`, `val_history.json`, `train_curve.png`, `val_curve.png`, `train_meta.json`, `checkpoints/*.pt`.
- `eval/<checkpoint>/report.json` with `mean.<setting>.<model>.hole_fraction` and `lpips_occluded` for `apple_768`, `apple_decoder_<grid>` and ours. Same comparison images and videos as stage 2.

## Gotchas

- The novel-view color loss is still masked. Only alpha and perceptual see the holes. A drop in `hole_fraction` and `lpips_occluded` is the signal that disocclusions are filling in; full-image LPIPS can lag behind.
- Depth correction is off by default in stage 3. Stage 2 trained with it, and run `stage3-b29749e` held that alignment frozen, so the decoder learned on ground-truth-corrected depth and was scored on raw monodepth. Compare a `--depth-correction 1` run against the default on the eval `inference` setting, not on `gt_depth`.
- `history.json` `grad_norm` is the pre-clip norm of the entry's last step. `grad_norm_mean`, `grad_norm_max` and `clipped_fraction` cover every step since the previous entry. `train_meta.json` records `grad_clip` and gets `peak_memory_gb` during the loop.
- The perceptual net is an ImageNet ResNet-50, baked into the Modal image. The paper's inpainting ResNet-50 is not used. See `docs/training-spec.md`.
- Checkpoints still exclude monodepth weights and inherit Apple's research-only model license. Renders are Hypersim: credit Hypersim (CC BY-SA 3.0).
