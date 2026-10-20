# Overfit

Reset the Gaussian decoder and the prediction head to their fresh initialization, then train on one pair. The loss must drop toward zero and the renders must converge.

## Sub-features

- Loss curve (`history.json`, `overfit_curve.png`).
- Renders before, during and after (`overfit_step_000000.png`, `snapshot_*.png`, `overfit_final.png`).

## How to get to it (user POV)

`stage1_report.json` → `overfit.before`, `overfit.after`, `overfit.criteria`.

## Driving it with modal

`--overfit-steps` (default 300) and `--overfit-lr` (default 5e-4). Novel-view losses use the frustum mask.

## Gotchas

- The fresh head outputs zeros, so the first render is the initializer's base Gaussians: input colors at the aligned depth, opacity 0.5. The start loss is therefore moderate, not huge.
