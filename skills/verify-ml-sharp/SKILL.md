---
name: verify-ml-sharp
description: Verify the SHARP training loop in DGFX/ml-sharp on Modal. Surface is the Modal app `ml-sharp-train` driven by `modal run modal_app.py`; it runs the stage 1 checks (gradient flow, Apple-weights loss, single-pair overfit) and the stage 2 checks, training and per-checkpoint eval on real Hypersim pairs. Use after any change to src/sharp_train, src/sharp, modal_app.py or scripts/prepare_hypersim.py, before merging, or when asked to verify ml-sharp.
---

# Verify ml-sharp

The product is a training loop. A user drives it with `modal run` and judges it by loss numbers and renders. Proof means the real model on real Hypersim pairs on a real GPU. There are no mocks.

## Launch

Run from a machine with Modal auth (Dominik's Mac or ai-vps, `~/.modal.toml`). Never print the token or put it on a command line.

1. Data on the volume. If `modal volume ls ml-sharp-data hypersim` does not list `pairs.json`, prepare and upload from ai-vps:
   ```
   uv run --with h5py --with remotezip --with pillow --with numpy python scripts/prepare_hypersim.py --out ~/data/hypersim --scenes ai_005_001 ai_007_001 ai_001_001 ai_001_002 ai_002_001 ai_003_001 ai_004_001 ai_006_001 ai_008_001 ai_009_001 ai_010_001 ai_011_001 --frames 40
   modal volume put ml-sharp-data ~/data/hypersim /hypersim
   ```
2. Spend before: `modal billing report --start $(date +%F) --json`.
3. Stage 1:
   ```
   modal run modal_app.py --task stage1 --run-name stage1-$(git rev-parse --short HEAD) --eval-scenes ai_005_001,ai_007_001
   ```
   The first run builds the image (gsplat compiles its CUDA kernels, CPU only, about 10 to 20 minutes, no GPU billed). Ready means the log prints the `gradient_check` JSON line. The GPU part takes about 10 minutes on an A100-40GB, timeout 45 minutes. The Apple checkpoint is cached on the volume at `/vol/checkpoints`.

## Doctor

Read-only, run first when anything looks off:

```
modal profile current
modal app list | grep ml-sharp-train
modal volume ls ml-sharp-data hypersim | head
modal volume ls ml-sharp-data checkpoints
```

The app must not be running from someone else's session. `pairs.json` and at least one scene folder must exist.

## Drive

Each feature in [features/](features/README.md) is one function of `modal_app.py`, reached through `--task`:

| Task | Feature |
| --- | --- |
| `stage1` | [gradient flow](features/gradient-flow.md), [Apple-weights check](features/apple-weights.md), [overfit](features/overfit.md) |
| `stage2-checks`, `stage2-train`, `stage2-eval` | [stage 2 smoke](features/stage2-smoke.md) |

## Evidence

Every run writes to `/vol/runs/<run-name>/` on the volume. Fetch it:

```
modal volume get ml-sharp-data runs/<run-name> ./runs/
```

- `stage1_report.json`: every number and every criterion, `"passed"` at the top.
- `apple_<pair>.png`: rows are input view and novel view; columns are ground truth, render, GT disparity, rendered disparity.
- `overfit_step_000000.png`, `snapshot_*.png`, `overfit_final.png`, `overfit_curve.png`, `history.json`.

Proof standard: read the report and open the images. A pass in JSON with renders that do not look like the targets is a failure. Put the report numbers, the spend delta and the image paths in the PR body.

## Cleanup

`modal run` stops the app when the function returns. Confirm with `modal app list` that `ml-sharp-train` is stopped. If a detached run is still going and you started it, `modal app stop ml-sharp-train`. Never delete `/vol/runs`: it is the evidence. `./runs/` locally is scratch.

## Helpers

None beyond `modal_app.py` and `scripts/prepare_hypersim.py`. The checks live in `src/sharp_train/checks.py`.
