# Stage 2 smoke

Short training run with a 192 x 192 Gaussian grid (instead of 768), visibility mask on the novel view, then LPIPS and DISTS on held-out pairs against Apple's weights and a camera-move video.

## Sub-features

- `stage2-train`: decoder and head reset, monodepth frozen from Apple's checkpoint, Adam with warmup and cosine decay, wall-clock cap `--max-minutes`.
- `stage2-eval`: novel-view LPIPS (AlexNet) and DISTS on full images, PSNR and L1 inside the visibility mask, inference setting (no GT depth). Video `eval/camera_move.mp4`, Apple on the left, ours on the right.

## How to get to it (user POV)

```
modal run --detach modal_app.py --task stage2-train --run-name stage2-<sha> --train-scenes <comma list> --eval-scenes ai_005_001,ai_007_001 --steps 2000 --max-minutes 25
modal run modal_app.py --task stage2-eval --run-name stage2-<sha> --eval-scenes ai_005_001,ai_007_001
```

## Driving it with modal

`--grid 192`, `--lr`, `--steps`, `--max-minutes`, `--num-eval-pairs`, `--video-pair`.

## Gotchas

- Expect our numbers to be far worse than Apple's: a few thousand steps on a handful of scenes against 160K steps on 8M images. The smoke proves the loss falls and the pipeline runs end to end.
- `decoder.pt` excludes monodepth weights; `stage2-eval` rebuilds monodepth from Apple's checkpoint.
