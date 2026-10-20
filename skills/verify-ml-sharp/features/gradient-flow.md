# Gradient flow

Upstream SHARP was written for inference. This check proves the training forward is differentiable end to end.

## Sub-features

- Static scan of `src/sharp` forward-path files for `no_grad`, `inference_mode`, `.eval()`, `.detach()`, `requires_grad_(False)` and `.numpy()`. Listed in `static_scan`, for reading, not for pass or fail.
- Runtime check: one forward and backward on a real pair. Gradient on the prediction head output per attribute group.
- Contrast run through upstream `unproject_gaussians`.

## How to get to it (user POV)

`modal run modal_app.py --task stage1 ...`, first JSON line of the log, and `gradient_check` in `stage1_report.json`.

## Driving it with modal

No extra flags. It uses the first Apple-check pair.

## Gotchas

- `delta_grad_apple_unproject.scale` and `.quaternion` are 0 by design of upstream code (`decompose_covariance_matrices` detaches and runs SVD on the CPU). That is expected and is why `sharp_train.model.to_metric` exists. If `delta_grad_ours` shows a 0 anywhere, the training forward is broken.
- The monodepth backbone is frozen by `create_monodepth_dpt` (`monodepth_trainable_params` is 0). That matches the paper's default and is not a bug.
