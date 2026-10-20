# Training spec from the paper

Source: `docs/paper/sharp-2512.10685.pdf`, Section 3.3 and 3.4, Appendix A.

## Inputs and outputs

- Input image resized to 1536 x 1536. Monodepth (Depth Pro style, two depth layers) predicts disparity. During training a learned depth-alignment U-Net aligns monodepth to ground-truth depth. At inference the alignment is the identity.
- Two Gaussian layers on a 768 x 768 grid: 1,179,648 Gaussians. Colors are linear RGB; the renderer converts to sRGB after compositing.
- Each sample is an input view and a novel view. Gaussians are predicted from the input view and rendered in both views.

## Losses (Eq. 3.12) and weights (Appendix A.2)

| Term | Definition | Weight | Here |
| --- | --- | --- | --- |
| color | L1 on rendered color, input and novel view | 1.0 | stage 1 |
| depth | L1 on disparity, input view, first depth layer | 0.2 | stage 1 uses rendered disparity on both views, weight 0.2 |
| alpha | BCE of rendered alpha against 1, both views | 1.0 | stage 3, full novel view |
| percep | feature loss + Gram loss on a ResNet-50, novel view, weights `1/(D H W)` and `10/D^2` | 3.0 | stage 3, full novel view |
| tv | total variation on the second depth layer | 1.0 | stage 3 |
| grad | floater penalty on large disparity gradients, sigma = eps = 1e-2 | 0.5 | stage 3 |
| delta | hinge on xy offsets beyond 400 | 1.0 | stage 3 |
| splat | hinge on projected variance outside [1e-1, 1e2] | 1.0 | stage 3 |
| scale | MAE of the alignment scale map to 1 | 0.1 | stage 3, only when depth correction is on |
| grad scale | multi-scale TV on the scale map, 6 levels | 5.0 | stage 3, only when depth correction is on |

## Masking (Appendix A.3)

Novel-view pixels are back-projected with novel depth into the input camera. Supervision only where the projection lands inside [-1.05, 1.05] in NDC. Applied to every image loss on the novel view. `geometry.frustum_mask` implements this.

Stage 2 trains with `geometry.visibility_mask` instead: a novel-view pixel counts only if its ground-truth depth, reprojected into the input camera, lands inside the input frame and agrees with the input view's ground-truth depth there within 5 %. Pixels hidden behind something in the input view are dropped, so the loss never asks the model to invent content it could not see. The mask applies to the novel-view color and disparity terms. `checks.visibility_check` proves it on the 26 eval pairs: the mask is always inside the frustum, and warping the input image into the novel view with ground-truth depth has 3x lower error inside the mask (0.043) than on the occluded pixels it removes (0.143).

Stage 3 keeps that mask on novel-view color and disparity. Alpha BCE and the perceptual loss are computed on the full novel view, with no visibility mask and no frustum mask, so disocclusions and the frame edge get a gradient. That is a deliberate departure from Appendix A.3, which applies the frustum mask to every image loss on the novel view. Input-view alpha is unmasked, as in the paper.

## Gaussian grid and input resolution (stage 2)

Apple's model runs the frozen Depth Pro monodepth at 1536 x 1536 and predicts two Gaussians per cell of a 768 x 768 grid. `sharp_train.model.GridPredictor(predictor, grid)` makes the grid configurable: with `factor = 768 / grid`, the monodepth still runs at 1536 (its ViT patching is fixed to that size), then its disparity, encoder features and decoder features are average-pooled by `factor`, and the image goes into the initializer at `1536 / factor`. Everything trainable (depth alignment, initializer, Gaussian decoder, head) runs at the reduced resolution, so grid 192 means input 384, 73,728 Gaussians, 16x less decoder compute. `checks.grid_equivalence_check` proves the wrapper at grid 768 gives bit-identical Gaussians to Apple's forward.

## Optimisation (Appendix A.2)

Adam, linear warmup for 10K steps to 1.6e-4, cosine decay to 1.6e-5. 100K steps on 128 A100s on synthetic data, then 60K steps of self-supervised fine-tuning on 32 A100s. FP32 throughout; the perceptual loss is computed with eager gradient pre-computation to fit 40 GB at batch size 1 (Appendix A.4).

## Activations (Appendix A.1)

| Attribute | Activation | Delta factor |
| --- | --- | --- |
| x/z, y/z | identity | 1e-3 |
| 1/z | softplus | 1e-3 |
| color | sigmoid | 1e-1 |
| rotation | identity | 1 |
| scale | sigmoid | 1 |
| alpha | sigmoid | 1 |

These are already in `src/sharp/models/composer.py` and `params.py`.

## Evaluation (stage 2)

`src/sharp_train/eval_pairs.json` is the fixed eval set: 26 pairs, 13 from each held-out scene (`ai_005_001`, `ai_007_001`), spread evenly over the camera baseline (0.12 m to 1.5 m), one per source frame. `scripts/pick_eval_pairs.py` regenerates it from `pairs.json`. It also lists 8 validation pairs used for the held-out loss during training and 2 video pairs.

`stage2.evaluate` reports, per pair and averaged, for every checkpoint and always for Apple's weights next to it:

| Metric | View | Region |
| --- | --- | --- |
| LPIPS (AlexNet), DISTS | novel | full image |
| PSNR, color L1 | novel | visibility mask |
| color L1 | input | full image |
| disparity L1 | input | valid GT depth |
| disparity L1 | novel | valid GT depth inside the visibility mask |
| hole fraction | novel | full image, share of pixels with rendered alpha < 0.5 |
| LPIPS (AlexNet) | novel | occluded region: the inverse of the visibility mask |

Each metric is computed in two settings: `inference` (no ground-truth depth, the setting a user gets) and `gt_depth` (monodepth aligned to ground truth, the training setting and the one stage 1 numbers use).

Occluded LPIPS composites the render with the ground truth outside the occluded region, then runs LPIPS against the ground truth. Pixels the visibility mask keeps contribute nothing, so the score moves only when the disoccluded region is wrong.

## Stage 3

`modal run modal_app.py --task stage3 --mode check|train|eval` continues from a checkpoint at the same grid. `--init stage2` (the default) loads `/vol/runs/stage2-29fefbe/checkpoints/final.pt`. `--init stage3b` loads `/vol/runs/stage3-b29749e/checkpoints/final.pt`, the held-alignment run whose decoder already overshot color and holes. A non-empty `--init-checkpoint` wins over `--init`. The Gaussian decoder is not reset. Monodepth stays frozen. Adam is fresh. Stage 3 warmup is `--stage3-warmup`, default 10,000, the paper's linear ramp to 1.6e-4. `--warmup` (default 300) is only the stage 2 schedule and is not read by stage 3. `--depth-correction 0` skips the scale map and the two scale losses. `--gpu` or `STAGE3_GPU` selects the GPU (`A100-40GB` default, `A100-80GB` when the perceptual loss does not fit). `--timeout-minutes` overrides the function timeout. `--max-minutes` is the budget inside the train loop.

Color and disparity stay the stage 1/2 rendered L1 (both views, novel view inside the visibility mask, depth weight 0.2). That is not the paper's depth loss, which is input-view, first depth layer only. On top of that total, stage 3 adds the weighted terms below. Logged values are unweighted; the weights go into `total`.

| Term | Implementation |
| --- | --- |
| alpha | `BCE(alpha, 1)` on the input view and the full novel view. Probabilities are clamped to `[1e-4, 1-1e-4]`. |
| percep | Eq. 3.4 on the full novel view only. Four ResNet-50 stages (`layer1` through `layer4`). Feature term is the mean square error (`λ_feat = 1/(D H W)`, Appendix A.2's `k = 1`). Gram term is `λ_Gram = 10/D^2` times the sum of squares of `φ φᵀ / (H W)` (`j = 10`). The backbone is torchvision `ResNet50_Weights.IMAGENET1K_V2` with ImageNet normalization (mean 0.485, 0.456, 0.406; std 0.229, 0.224, 0.225). The paper's ResNet-50 is pretrained for inpainting; those weights are not on public PyPI, so the ImageNet norm stays, because it is the norm those weights were trained with. Appendix A.4's eager autograd drops the ResNet graph after precomputing the gradient with respect to the rendered image. There is no ramp on `λ_percep`; Appendix A.2 fixes it at 3 for the whole run. |
| tv | Mean absolute horizontal difference plus mean absolute vertical difference of the inverse of the aligned second depth layer. |
| grad | Opacity times `1 - exp(-relu(|∇D̄⁻¹| - 1e-2) / 1e-2)`. The gradient is the per-pixel magnitude of the first depth layer's inverse depth, sampled at the base Gaussian's NDC position (`G0`, before the xy offset), for every Gaussian in both layers. |
| delta | `mean(relu(|Δx| - 400) + relu(|Δy| - 400))` on the raw xy channels of the prediction head, before the `1e-3` delta factor. A threshold of 400 in NDC would cover the whole image many times over; 400 on the raw head output is 0.4 in NDC after the factor from Appendix A.1. |
| splat | Hinge of the source-view projected variance outside `[0.1, 100]`. Variance is the mean eigenvalue of the 2D covariance (trace / 2) plus the renderer's `1e-2` low-pass on the diagonal. |
| scale | Mean absolute error of the depth-correction scale map to 1. The map is identically 1 when correction is off, during inference, or at the module's zero init. |
| grad scale | Sum over six levels of the same total-variation as `tv`, on the scale map average-pooled by `2^k`, `k = 1..6`. |

Depth correction is Apple's existing `DepthAlignment` / `LearnedAlignment` U-Net: predicted inverse depth and ground-truth inverse depth in, one scale map out, multiplied onto both depth layers before the Gaussian initializer. `align_depth(..., enabled)` is the switch. Enabled feeds ground truth whenever the training step has it. Disabled passes no ground truth, so the module returns the monodepth unchanged and a scale of 1. Inference still passes no ground truth, which is the paper's identity replacement.

A freshly constructed alignment is the identity: `conv_out` weight 0, bias `log 1`, so the scale is 1 and the depth is unchanged. The xy delta hinge is on the raw prediction-head offsets and is 0 while those offsets stay inside 400. Stage 2 then trains a non-identity scale. On the stage 2 checkpoint that scale already matches ground truth (source disparity about 0.002) while `|S-1|` is about 0.35 on the val set. Stage 3 does not reset it.

The scale losses keep the paper weights, but stage 3 does not train the alignment module. `train` calls `hold_depth_alignment` whenever stage 3 runs with depth correction, which freezes `depth_alignment`. The second-layer TV and the scale regularizers both read that shared scale. Left trainable, one Adam step moves a scale that reconstructs ground truth off the ground truth: second-layer TV pushes `|S-1|` up, and multiscale scale TV (weight 5) flattens spatial detail. The parameter gradient of the scale TV is orders of magnitude larger than a direct disparity loss, and the rendered disparity term (weight 0.2, value about 0.002) does not hold the map. The 8000-step run showed this in the first 150 steps, while the xy offset hinge was still 0: val `|S-1|` rose and target disparity went from 0.0017 to about 0.05. Freezing the module keeps the stage 2 scale, so ground-truth-depth color and disparity stay on that solution while the Gaussian decoder trains. `--depth-correction 0` skips the scale map and does not freeze the module.

Global gradient clipping is `clip_grad_norm_` at 1.0 (`GRAD_CLIP_NORM`). The paper does not name a clip. `history.json` records `grad_norm` before the clip, which is why a run can print 150–700 (and a spike near 3400) while the update is scaled to norm 1. Perceptual weight 3 makes that term most of the 19–33 total (about 30 of 33 at step 0, about 18 of 19 later). That ratio is the paper formula with these weights, not a missing factor of 100: `λ_feat` and `λ_Gram` already match Appendix A.2. `train` rewrites `peak_memory_gb` into `train_meta.json` on each log and before each callback, because a detached Modal run drops the function's return value.

The learning rate is Appendix A.2: linear warmup over 10,000 steps from `peak/10000` to 1.6e-4, then cosine decay to 1.6e-5 (`final_ratio` 0.1) across the remaining steps of the run. `losses.cosine_lr` is that curve. A run shorter than the warmup stays on the ramp the whole time, so 4,000 steps ends at `4000/10000 * 1.6e-4 = 6.4e-5`. The held run `stage3-b29749e` used the stage 2 warmup of 300 instead. Its `history.json` shows step 0 at `1.6e-4/300` and step 499 already on the cosine, above `0.9 * 1.6e-4`, while the weighted perceptual term was still about 30 of 33. Val target color L1 went from 0.054 at step 0 to 0.086 at step 500 and only recovered to 0.066. Ground-truth eval color L1 was 0.063 (stage 2 was 0.053) and PSNR 20.9 (stage 2 was 22.5). The next stage 3 run starts from `stage2`, not from `stage3-b29749e`, so it does not continue that overshot decoder.

Alpha BCE is Eq. 3.5: target 1 on the input view and on the full novel view, unmasked. Raising novel-view alpha lowers that term. On the held run it fell from 1.75 to 0.26, so the novel view was supervised, and a mean that low still leaves a tail of pixels under 0.5 (hole fraction about 0.07). The hole regression against the unheld run is not a wrong alpha target. Holding the scale removed the screen-space splat bloat that a drifting scale had been using to cover gaps (unheld inference holes 0.061, held 0.075; Apple at grid 192 is 0.072). With the short warmup, perceptual steps at full learning rate pushed projected variance above 100: val splat loss hit 0.48 at step 500, and the lower hinge cannot exceed 0.1, so 0.48 is the upper hinge (splats too big). The hinge then shrinks them and the gaps reopen. The offset hinge stayed 0 for the first ~1000 steps, because its dead zone is 400 on the raw head (0.4 NDC after the 1e-3 factor). The floater penalty stayed near 0.018 because the held depth edges do not move. Appendix A.4 describes this same perceptual failure mode: large, blobby Gaussians. The paper's counter is the Gram term, which is already on, and the 10k warmup, which this schedule restores. We do not retune `σ` from variance to standard deviation: Eq. 3.10 and Appendix A.2 call it variance, with `σ_min = 0.1` and `σ_max = 100`.

Perceptual loss stays on the novel view only (Eq. 3.4). Appendix A.4 says the released recipe also applies it to the input view. A second ResNet pass would raise the 0.38 s/step cost, so this repo keeps the equation's novel-view-only form. No perceptual-weight warmup is added; the paper does not have one.

## Data and license

Training and evaluation use a subset of [Hypersim](https://github.com/apple/ml-hypersim) (Roberts et al., "Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding", ICCV 2021). The Hypersim dataset is released under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); its code is under Apple's sample-code license. Renders, images and videos made from Hypersim scenes must credit Hypersim, and redistributed derived data must stay under CC BY-SA 3.0. The data is never committed; it lives on the Modal volume. Weights trained from Apple's checkpoint stay under [LICENSE_MODEL](../LICENSE_MODEL) (research use only).
