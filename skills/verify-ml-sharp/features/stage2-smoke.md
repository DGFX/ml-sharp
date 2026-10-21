# Stage 2 smoke

Short training run with a 192 x 192 Gaussian grid (instead of 768) and matching 384 input resolution, visibility mask on the novel view, then a per-checkpoint eval against Apple's weights: LPIPS, DISTS, color L1 and disparity L1 on the fixed held-out eval set, comparison images, camera-move videos.

## Sub-features

- `stage2-checks`: visibility mask on the 26 eval pairs (inside the frustum, lower warp error than the occluded pixels it removes), grid 768 wrapper is bit-identical to Apple's forward, grid 192 forward gives 73,728 Gaussians and a sane render with Apple's decoder, gradient flow through the wrapper, the reset head starts at zero and gets gradient, steady-state seconds per step and peak memory.
- `stage2-train`: Apple checkpoint with the Gaussian decoder and head reset, monodepth frozen, Adam with warmup and cosine decay, loss on the novel view inside the visibility mask, wall-clock cap `--max-minutes`. Held-out validation loss on 8 pairs every `--val-every` steps, checkpoints every `--checkpoint-every` steps and at the end.
- `stage2-eval`: for each checkpoint in `--checkpoints`, three models on the 26 eval pairs: `apple_768` (Apple as released), `apple_decoder_192` (Apple's weights through the 192 grid, no training) and ours. Two videos per checkpoint for Apple and ours.

## How to get to it (user POV)

```
modal run modal_app.py --task stage2-checks --run-name stage2-checks-<sha> --max-usd 0.93
modal run --detach modal_app.py --task stage2-train --run-name stage2-<sha> --steps 8000 --max-minutes 52 --max-usd 2.97
modal run modal_app.py --task stage2-eval --run-name stage2-<sha> --checkpoints step_004000,final --max-usd 1.49
```

## Driving it with modal

`--grid` (divisor of 768), `--lr`, `--warmup`, `--steps`, `--max-minutes`, `--checkpoint-every`, `--val-every`, `--seed`, `--train-scenes` (default: every prepared scene except the held-out ones), `--throughput-steps`, `--checkpoints` (comma list of names under `runs/<run>/checkpoints/`).

## Evidence

- `stage2_checks.json` with `"passed"`.
- `history.json` (training loss, window means), `val_history.json`, `train_curve.png`, `val_curve.png`, `train_meta.json`, `checkpoints/*.pt`.
- `eval/<checkpoint>/report.json`: `mean.<setting>.<model>` and `per_pair`. `compare_<pair>.png`: top row input, target, then each model's novel view; bottom row visibility mask, target disparity, each model's disparity. `video_<pair>_<model>.mp4`, `video_<pair>_side_by_side.mp4` (Apple left, ours right) and a small copy `_side_by_side_small.mp4`.

## Gotchas

- The reset head is zero-initialised, so the first step gives the decoder no gradient. Gradient flow is checked through the grid wrapper with Apple's decoder instead.
- `decoder.fusions.4.resnet1` never gets gradient. That is by design: the lowest fusion block has no skip input, so its skip branch is unused, in Apple's model too.
- Expect our numbers to be worse than Apple's: a few thousand steps on 10 scenes at a quarter of the resolution against 160K steps on 8M images. `apple_decoder_192` separates the cost of the smaller grid from the cost of training.
- Checkpoints exclude monodepth weights; loading rebuilds monodepth from Apple's checkpoint. They inherit Apple's research-only model license.
- Renders and videos come from Hypersim scenes: credit Hypersim (CC BY-SA 3.0) when sharing them.
