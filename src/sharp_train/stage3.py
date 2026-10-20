import json
import math

import torch

from sharp_train import losses


class Stage3Loss:
    def __init__(self, device, percep_weight=losses.STAGE3_WEIGHTS["percep"]):
        self.perceptual = losses.PerceptualLoss().to(device)
        self.weights = dict(losses.STAGE3_WEIGHTS)
        self.weights["percep"] = percep_weight


def _backward(loss, leaf):
    leaf.grad = None
    loss.backward()
    grad = leaf.grad
    return {
        "loss": float(loss.detach()),
        "finite": bool(
            torch.isfinite(loss).all() and grad is not None and torch.isfinite(grad).all()
        ),
        "grad": float(grad.abs().mean()) if grad is not None else 0.0,
    }


def _regularizer_checks():
    results = {}
    depth = torch.linspace(0.4, 2.0, 32, dtype=torch.float32).view(1, 1, 1, 32).expand(1, 2, 32, 32)
    depth = depth.clone().requires_grad_(True)
    tv = losses.total_variation(1.0 / depth[:, 1:2].clamp(min=1e-4))
    results["tv"] = _backward(tv, depth)

    inverse = (1.0 / depth[:, :1].detach().clamp(min=1e-4)).requires_grad_(True)
    layers, size = 2, 8
    coords = torch.linspace(-0.75, 0.75, size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    base_x = grid_x.view(1, 1, 1, size, size).expand(1, 1, layers, size, size).contiguous()
    base_y = grid_y.view(1, 1, 1, size, size).expand(1, 1, layers, size, size).contiguous()
    opacities = torch.full((1, layers, size, size), 0.8, requires_grad=True)
    floater = losses.floater_loss(opacities, inverse, base_x, base_y)
    results["grad"] = _backward(floater, opacities)

    delta = torch.zeros(1, 2, layers, size, size)
    delta[0, 0, 0, 0, 0] = 500.0
    delta.requires_grad_(True)
    results["delta"] = _backward(losses.offset_hinge(delta), delta)

    means = torch.zeros(1, 1, 3)
    means[0, 0, 2] = 2.0
    eye = torch.eye(3).view(1, 1, 3, 3)
    intrinsics = torch.tensor([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]).view(
        1, 3, 3
    )
    scale = 0.02
    variance = losses.projected_variance(means, eye * scale**2, intrinsics)
    expected = (100.0 / 2.0 * scale) ** 2 + losses.SPLAT_LOW_PASS
    identity = {
        "variance": float(variance),
        "expected": expected,
        "matches": bool(torch.allclose(variance, torch.tensor(expected), rtol=1e-4, atol=1e-4)),
    }
    big = (eye * 4.0).clone().requires_grad_(True)
    splat = losses.splat_hinge(losses.projected_variance(means, big, intrinsics))
    results["splat"] = _backward(splat, big)
    results["splat_identity"] = identity

    scale_map = (torch.rand(1, 1, 64, 64) * 0.4 + 0.8).requires_grad_(True)
    results["scale"] = _backward(losses.scale_mae(scale_map), scale_map)
    scale_map2 = scale_map.detach().clone().requires_grad_(True)
    results["grad_scale"] = _backward(losses.multiscale_tv(scale_map2), scale_map2)

    alpha = torch.full((1, 1, 8, 8), 0.4, requires_grad=True)
    results["alpha"] = _backward(losses.alpha_bce(alpha), alpha)
    return results


def _hole_fraction_check():
    alpha = torch.tensor([0.0, 0.49, 0.5, 0.9], dtype=torch.float32).view(1, 1, 1, 4)
    value = float(losses.hole_fraction(alpha))
    inside = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32).view(1, 1, 1, 4)
    masked = float(losses.hole_fraction(alpha, mask=inside))
    return {
        "hole_fraction": value,
        "expected": 0.5,
        "matches": math.isclose(value, 0.5),
        "frustum_masked": masked,
        "frustum_masked_differs": not math.isclose(value, masked) and math.isclose(masked, 0.0),
    }


def _fresh_alignment():
    from sharp.models.alignment import LearnedAlignment
    from sharp.models.predictor import DepthAlignment

    estimator = LearnedAlignment(steps=4, stride=1, base_width=16, depth_decoder_features=False)
    return DepthAlignment(estimator), estimator


def _disparity_error(alignment, monodepth, gt_depth):
    from sharp_train.model import align_depth

    with torch.no_grad():
        corrected, _ = align_depth(alignment, monodepth, gt_depth, None, True)
        error = losses.disparity_l1(corrected[:, :1], gt_depth, torch.ones_like(gt_depth))
    return float(error), corrected


def _base_metric_depth(corrected):
    from sharp.models.initializer import create_initializer
    from sharp.models.params import InitializerParams

    image = torch.zeros(1, 3, *corrected.shape[-2:])
    output = create_initializer(InitializerParams())(image, corrected)
    inverse = output.gaussian_base_values.mean_inverse_z_ndc[:, 0, 0]
    return output.global_scale.view(-1, 1, 1) / inverse.clamp(min=1e-4)


def _gt_aligned_pair():
    alignment, estimator = _fresh_alignment()
    generator = torch.Generator().manual_seed(0)
    gt_depth = torch.rand(1, 1, 32, 32, generator=generator) * 1.5 + 0.8
    estimator.conv_out.weight.data.zero_()
    estimator.conv_out.bias.data.fill_(1.0)
    first = gt_depth / math.e
    second = first * (0.55 + 0.9 * torch.rand(1, 1, 32, 32, generator=generator))
    monodepth = torch.cat([first, second], dim=1)
    return alignment, monodepth, gt_depth


def _regularizer_step(alignment, monodepth, gt_depth):
    from sharp_train.model import align_depth

    parameters = [parameter for parameter in alignment.parameters() if parameter.requires_grad]
    if not parameters:
        return
    optimizer = torch.optim.Adam(parameters, lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    corrected, scale = align_depth(alignment, monodepth, gt_depth, None, True)
    inverse = 1.0 / corrected.clamp(min=1e-4)
    loss = (
        losses.STAGE3_WEIGHTS["scale"] * losses.scale_mae(scale)
        + losses.STAGE3_WEIGHTS["grad_scale"] * losses.multiscale_tv(scale)
        + losses.STAGE3_WEIGHTS["tv"] * losses.total_variation(inverse[:, 1:2])
    )
    loss.backward()
    losses.clip_gradients(parameters)
    optimizer.step()


def _depth_correction_check():
    from sharp_train.model import align_depth, hold_depth_alignment

    alignment, estimator = _fresh_alignment()
    monodepth = torch.rand(1, 2, 32, 32) + 0.5
    gt_depth = torch.rand(1, 1, 32, 32) + 0.5
    initial, _ = align_depth(alignment, monodepth, gt_depth, None, True)
    estimator.conv_out.bias.data.fill_(1.0)
    enabled, scale = align_depth(alignment, monodepth, gt_depth, None, True)
    disabled, disabled_scale = align_depth(alignment, monodepth, gt_depth, None, False)

    open_alignment, open_mono, open_gt = _gt_aligned_pair()
    open_before, _ = _disparity_error(open_alignment, open_mono, open_gt)
    _regularizer_step(open_alignment, open_mono, open_gt)
    open_after, _ = _disparity_error(open_alignment, open_mono, open_gt)

    held_alignment, held_mono, held_gt = _gt_aligned_pair()
    held_before, held_corrected = _disparity_error(held_alignment, held_mono, held_gt)
    held_depth = _base_metric_depth(held_corrected)
    gt_disparity = torch.nn.functional.max_pool2d(1.0 / held_gt, 2)
    hold_depth_alignment(held_alignment)
    _regularizer_step(held_alignment, held_mono, held_gt)
    held_after, held_corrected = _disparity_error(held_alignment, held_mono, held_gt)
    held_depth_after = _base_metric_depth(held_corrected)
    return {
        "init_is_identity": bool(torch.allclose(initial, monodepth, atol=1e-5)),
        "enabled_matches_exp_scale": bool(
            torch.allclose(enabled, monodepth * math.e, rtol=1e-4, atol=1e-4)
        ),
        "enabled_changes_depth": bool((enabled - monodepth).abs().mean() > 1e-3),
        "disabled_keeps_depth": bool(torch.equal(disabled, monodepth)),
        "disabled_scale_is_one": bool(torch.equal(disabled_scale, torch.ones_like(disabled_scale))),
        "enabled_scale_away_from_one": bool((scale - 1).abs().mean() > 1e-3),
        "open_error_before": open_before,
        "open_error_after": open_after,
        "regularizer_increases_gt_error": bool(open_after > open_before + 1e-3),
        "held_error_before": held_before,
        "held_error_after": held_after,
        "held_keeps_gt_depth": bool(math.isclose(held_before, held_after, abs_tol=1e-8)),
        "base_depth_matches_gt": bool(
            torch.allclose(held_depth, 1.0 / gt_disparity, rtol=1e-4, atol=1e-5)
        ),
        "base_depth_unchanged": bool(torch.allclose(held_depth, held_depth_after)),
    }


def _loss_scale_check():
    paper = {
        "alpha": 1.0,
        "percep": 3.0,
        "tv": 1.0,
        "grad": 0.5,
        "delta": 1.0,
        "splat": 1.0,
        "scale": 0.1,
        "grad_scale": 5.0,
    }

    def total(percep, alpha, grad_scale, scale, color, delta=0.0, splat=0.0):
        extra = {
            "alpha": torch.tensor(alpha),
            "percep": torch.tensor(percep),
            "tv": torch.tensor(0.002),
            "grad": torch.tensor(0.02),
            "delta": torch.tensor(delta),
            "splat": torch.tensor(splat),
            "scale": torch.tensor(scale),
            "grad_scale": torch.tensor(grad_scale),
        }
        terms = losses.add_stage3({"total": torch.tensor(color)}, extra)
        weighted_percep = losses.STAGE3_WEIGHTS["percep"] * percep
        return float(terms["total"]), weighted_percep

    early_total, early_percep = total(9.87, 1.75, 0.358, 0.347, 0.077)
    late_total, late_percep = total(6.05, 0.222, 0.014, 0.842, 0.138, delta=0.140, splat=0.047)
    leaf = torch.nn.Parameter(torch.zeros(4, 8))
    leaf.grad = torch.full((4, 8), 50.0)
    preclip = float(losses.clip_gradients([leaf]))
    postclip = float(leaf.grad.norm())
    return {
        "paper_weights": losses.STAGE3_WEIGHTS == paper,
        "early_total": early_total,
        "early_percep": early_percep,
        "early_percep_dominates": bool(early_percep > 0.6 * early_total),
        "late_total": late_total,
        "late_percep": late_percep,
        "late_percep_dominates": bool(late_percep > 0.6 * late_total),
        "grad_clip": losses.GRAD_CLIP_NORM,
        "preclip": preclip,
        "postclip": postclip,
        "clips_to_one": bool(preclip > 1.0 and postclip <= 1.0 + 1e-5),
    }


def _view(color, alpha):
    from types import SimpleNamespace

    return SimpleNamespace(color=color, alpha=alpha, depth=torch.ones_like(alpha))


def _state():
    from sharp_train.model import TrainingState

    layers, size = 2, 8
    depth = torch.linspace(0.4, 2.0, 32).view(1, 1, 1, 32).expand(1, 2, 32, 32).clone()
    coords = torch.linspace(-0.75, 0.75, size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    return TrainingState(
        aligned_depth=depth,
        scale_map=torch.ones(1, 1, 32, 32),
        delta_xy=torch.zeros(1, 2, layers, size, size),
        base_x=grid_x.view(1, 1, 1, size, size).expand(1, 1, layers, size, size).contiguous(),
        base_y=grid_y.view(1, 1, 1, size, size).expand(1, 1, layers, size, size).contiguous(),
    )


def _gaussians():
    from types import SimpleNamespace

    count = 2 * 8 * 8
    means = torch.zeros(1, count, 3)
    means[:, :, 2] = 2.0
    covariances = torch.eye(3).view(1, 1, 3, 3).expand(1, count, 3, 3).contiguous() * 0.0004
    return SimpleNamespace(
        means=means, covariances=covariances, opacities=torch.full((1, count), 0.8)
    )


def _novel_pair(color, alpha, depth, batch, perceptual, state, gaussians):
    source = _view(color.detach(), alpha.detach())
    novel = _view(color, alpha)
    novel.depth = depth
    color_key = losses.resolve_color_mask_key(None, True)
    terms = losses.pair_losses(
        source,
        novel,
        batch,
        tgt_mask_key="mask_visible_tgt",
        color_mask_key=color_key,
    )
    extra = losses.stage3_terms(source, novel, batch, gaussians, state, perceptual)
    return terms, extra


def _region_grad(loss, leaf, columns):
    leaf.grad = None
    loss.backward(retain_graph=True)
    return float(leaf.grad[:, :, :, columns].abs().sum())


def _full_view_check(perceptual):
    torch.manual_seed(0)
    color = torch.rand(1, 3, 32, 32) * 0.4 + 0.4
    target = torch.rand(1, 3, 32, 32) * 0.2 + 0.1
    alpha = torch.full((1, 1, 32, 32), 0.7)
    depth = torch.ones(1, 1, 32, 32)
    frustum = torch.ones(1, 1, 32, 32)
    frustum[:, :, :, :8] = 0
    visible = frustum.clone()
    visible[:, :, :, :16] = 0
    outside = color.clone()
    outside[:, :, :, :8] = 0.02
    outside_alpha = alpha.clone()
    outside_alpha[:, :, :, :8] = 0.05
    outside_depth = depth.clone()
    outside_depth[:, :, :, :8] = 4.0
    state = _state()
    gaussians = _gaussians()
    intrinsics = torch.tensor([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]).view(
        1, 3, 3
    )
    batch = {
        "image_src": target,
        "image_tgt": target,
        "depth_src": depth,
        "depth_tgt": depth,
        "intrinsics": intrinsics,
        "mask_visible_tgt": visible,
        "mask_frustum_tgt": frustum,
    }
    with torch.no_grad():
        base_terms, base_extra = _novel_pair(
            color, alpha, depth, batch, perceptual, state, gaussians
        )
        changed_terms, changed_extra = _novel_pair(
            outside, outside_alpha, outside_depth, batch, perceptual, state, gaussians
        )
    color_leaf = outside.detach().clone().requires_grad_(True)
    alpha_leaf = outside_alpha.detach().clone().requires_grad_(True)
    depth_leaf = outside_depth.detach().clone().requires_grad_(True)
    terms, extra = _novel_pair(
        color_leaf, alpha_leaf, depth_leaf, batch, perceptual, state, gaussians
    )
    outside_cols = slice(0, 8)
    color_grad_outside = _region_grad(terms["color_tgt"], color_leaf, outside_cols)
    alpha_grad_outside = _region_grad(extra["alpha"], alpha_leaf, outside_cols)
    percep_grad_outside = _region_grad(extra["percep"], color_leaf, outside_cols)
    masked_color = color_leaf.detach().clone().requires_grad_(True)
    masked_alpha = alpha_leaf.detach().clone().requires_grad_(True)
    masked_color_grad = _region_grad(
        losses.color_l1(masked_color, target, frustum), masked_color, outside_cols
    )
    masked_alpha_grad = _region_grad(
        losses.alpha_bce(masked_alpha, frustum), masked_alpha, outside_cols
    )
    masked_percep_color = color_leaf.detach().clone().requires_grad_(True)
    masked_percep_grad = _region_grad(
        perceptual(losses.in_frustum(masked_percep_color, target, frustum), target),
        masked_percep_color,
        outside_cols,
    )
    direct = color.detach().clone().requires_grad_(True)
    perceptual.value(direct, target).backward()
    through_surgery = color.detach().clone().requires_grad_(True)
    perceptual(through_surgery, target).backward()
    return {
        "color_key": losses.resolve_color_mask_key(None, True),
        "color_changes_outside": bool(
            not torch.allclose(base_terms["color_tgt"], changed_terms["color_tgt"])
        ),
        "alpha_changes_outside": bool(
            not torch.allclose(base_extra["alpha"], changed_extra["alpha"])
        ),
        "percep_changes_outside": bool(
            not torch.allclose(base_extra["percep"], changed_extra["percep"])
        ),
        "depth_ignores_outside": bool(
            torch.allclose(base_terms["depth_tgt"], changed_terms["depth_tgt"])
        ),
        "color_grad_outside": color_grad_outside,
        "alpha_grad_outside": alpha_grad_outside,
        "percep_grad_outside": percep_grad_outside,
        "frustum_masked_color_grad_outside": masked_color_grad,
        "frustum_masked_alpha_grad_outside": masked_alpha_grad,
        "frustum_masked_percep_grad_outside": masked_percep_grad,
        "surgery_matches_backward": bool(
            torch.allclose(direct.grad, through_surgery.grad, rtol=1e-3, atol=1e-4)
        ),
        "perceptual_finite": bool(torch.isfinite(base_extra["percep"])),
        "wired_finite": all(bool(torch.isfinite(value).all()) for value in base_extra.values()),
        "frozen_depth_terms_are_zero": bool(
            float(base_extra["tv"]) == 0.0
            and float(base_extra["scale"]) == 0.0
            and float(base_extra["grad_scale"]) == 0.0
        ),
    }


def _schedule_check():
    peak = losses.PAPER_PEAK_LR
    warmup = losses.PAPER_WARMUP
    total = 100_000
    step0 = losses.cosine_lr(0, total, peak, warmup)
    step499 = losses.cosine_lr(499, total, peak, warmup)
    step9999 = losses.cosine_lr(warmup - 1, total, peak, warmup)
    step10000 = losses.cosine_lr(warmup, total, peak, warmup)
    end = losses.cosine_lr(total - 1, total, peak, warmup)
    short = losses.cosine_lr(499, 8000, peak, 300)
    probe = losses.cosine_lr(3999, 4000, peak, warmup)
    brief = losses.SHORT_RUN_WARMUP
    brief_end = losses.cosine_lr(brief - 1, 4000, peak, brief)
    brief_after = losses.cosine_lr(brief, 4000, peak, brief)
    brief_last = losses.cosine_lr(3999, 4000, peak, brief)
    return {
        "step0": step0,
        "step499": step499,
        "step9999": step9999,
        "step10000": step10000,
        "final": end,
        "warmup300_step499": short,
        "probe_step3999": probe,
        "starts_at_peak_over_warmup": math.isclose(step0, peak / warmup),
        "still_ramping_at_499": math.isclose(step499, peak * 500 / warmup) and step499 < 0.1 * peak,
        "reaches_peak_at_end_of_warmup": math.isclose(step9999, peak),
        "cosine_continuous_at_warmup": math.isclose(step10000, peak, rel_tol=1e-6),
        "cosine_ends_at_final": math.isclose(end, peak * losses.PAPER_FINAL_RATIO, rel_tol=1e-3),
        "short_warmup_at_peak_by_499": short > 0.9 * peak,
        "probe_stays_on_ramp": math.isclose(probe, peak * 4000 / warmup),
        "short_run_warmup": brief,
        "short_run_reaches_peak": math.isclose(brief_end, peak),
        "short_run_cosine_after_warmup": math.isclose(brief_after, peak, rel_tol=1e-6)
        and brief_last < 0.5 * peak,
    }


def _alpha_target_check(perceptual):
    color = torch.rand(1, 3, 32, 32) * 0.4 + 0.3
    src_alpha = torch.full((1, 1, 32, 32), 0.55)
    novel_alpha = torch.full((1, 1, 32, 32), 0.2)
    raised_alpha = torch.full((1, 1, 32, 32), 0.85)
    state = _state()
    gaussians = _gaussians()
    intrinsics = torch.tensor([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]).view(
        1, 3, 3
    )
    batch = {
        "image_tgt": color,
        "intrinsics": intrinsics,
        "mask_frustum_tgt": torch.ones(1, 1, 32, 32),
    }
    with torch.no_grad():
        base = losses.stage3_terms(
            _view(color, src_alpha),
            _view(color, novel_alpha),
            batch,
            gaussians,
            state,
            perceptual,
        )["alpha"]
        raised = losses.stage3_terms(
            _view(color, src_alpha),
            _view(color, raised_alpha),
            batch,
            gaussians,
            state,
            perceptual,
        )["alpha"]
    expected = losses.alpha_bce(src_alpha) + losses.alpha_bce(novel_alpha)
    manual = torch.nn.functional.binary_cross_entropy(
        novel_alpha.clamp(1e-4, 1.0 - 1e-4), torch.ones_like(novel_alpha)
    )
    return {
        "base": float(base),
        "raised": float(raised),
        "wired_to_both_views": bool(torch.allclose(base, expected)),
        "matches_bce_against_one": bool(torch.allclose(losses.alpha_bce(novel_alpha), manual)),
        "raising_novel_alpha_lowers_loss": bool(raised < base),
    }


def _hinge_check():
    zero = losses.splat_hinge(torch.zeros(4))
    upper = losses.splat_hinge(torch.tensor([losses.SPLAT_MAX + 0.48]))
    inside = losses.splat_hinge(torch.tensor([1.0]))
    at_limit = losses.offset_hinge(torch.full((3, 2), losses.OFFSET_LIMIT))
    at_zero = losses.offset_hinge(torch.zeros(3, 2))
    past = losses.offset_hinge(torch.full((3, 2), losses.OFFSET_LIMIT + 100))
    return {
        "zero_variance": float(zero),
        "upper": float(upper),
        "offset_past": float(past),
        "zero_variance_is_lower_cap": bool(torch.allclose(zero, torch.tensor(losses.SPLAT_MIN))),
        "upper_hinge_is_048": math.isclose(float(upper), 0.48, abs_tol=1e-5),
        "inside_band_is_zero": float(inside) == 0.0,
        "offset_dead_zone": float(at_limit) == 0.0 and float(at_zero) == 0.0,
        "offset_fires_past_400": float(past) > 0,
    }


def _perceptual_scale_check(perceptual):
    torch.manual_seed(1)
    rendered = torch.randn(2, 4, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    got = losses.feature_gram_loss(rendered, target)
    batch, channels, height, width = rendered.shape
    feature = (rendered - target).square().sum() / (channels * height * width * batch)

    def gram(value):
        matrix = value.flatten(2)
        return matrix @ matrix.transpose(1, 2) / (height * width)

    gram = 10.0 * (gram(rendered) - gram(target)).square().sum() / (channels * channels * batch)
    mean = torch.tensor(losses.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(losses.IMAGENET_STD).view(1, 3, 1, 1)
    image = torch.full((1, 3, 2, 2), 0.5)
    normalized = losses.imagenet_normalize(image, mean, std)
    constant = torch.full((1, 3, 32, 32), 0.5)
    with torch.no_grad():
        before = perceptual.features(constant)[-1].clone()
        saved_mean = perceptual.mean.clone()
        saved_std = perceptual.std.clone()
        perceptual.mean.zero_()
        perceptual.std.fill_(1)
        after = perceptual.features(constant)[-1]
        perceptual.mean.copy_(saved_mean)
        perceptual.std.copy_(saved_std)
    return {
        "feature_gram_matches_paper": bool(torch.allclose(got, feature + gram)),
        "identical_features_are_zero": bool(
            losses.feature_gram_loss(rendered, rendered).item() == 0.0
        ),
        "imagenet_mean_maps_to_zero": bool(
            torch.allclose(losses.imagenet_normalize(mean, mean, std), torch.zeros_like(mean))
        ),
        "half_uses_imagenet_stats": bool(torch.allclose(normalized, (0.5 - mean) / std)),
        "normalization_changes_features": bool(not torch.allclose(before, after)),
    }


def _composer():
    from sharp.models.composer import GaussianComposer
    from sharp.models.params import PredictorParams

    params = PredictorParams()
    return GaussianComposer(
        delta_factor=params.delta_factor,
        min_scale=params.min_scale,
        max_scale=params.max_scale,
        color_activation_type=params.color_activation_type,
        opacity_activation_type=params.opacity_activation_type,
        color_space=params.color_space,
        scale_factor=1,
        base_scale_on_predicted_mean=params.base_scale_on_predicted_mean,
    )


def _room_depth(size):
    y = torch.linspace(0, 1, size)
    x = torch.linspace(0, 1, size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    floor = 1.5 / (0.2 + 0.8 * (1.0 - yy).clamp(min=0.05))
    depth = torch.where(yy > 0.42, floor, torch.full_like(yy, 5.0))
    box = ((xx - 0.62).abs() < 0.1) & ((yy - 0.58).abs() < 0.18)
    return torch.where(box, torch.full_like(depth, 1.3), depth).view(1, 1, size, size)


def _projected_base(init_size, depth):
    from sharp.models.initializer import create_initializer
    from sharp.models.params import InitializerParams

    from sharp_train import geometry, model

    depth = torch.nn.functional.interpolate(
        depth, size=(init_size, init_size), mode="bilinear", align_corners=False
    ).repeat(1, 2, 1, 1)
    output = create_initializer(InitializerParams())(torch.zeros(1, 3, init_size, init_size), depth)
    base = output.gaussian_base_values
    grid = base.mean_x_ndc.shape[-1]
    delta = torch.zeros(1, 14, base.mean_inverse_z_ndc.shape[2], grid, grid)
    gaussians = _composer()(delta, base, output.global_scale)
    width, height, focal = 1024, 768, 886.0
    metric = model.to_metric(gaussians, torch.tensor([focal]), width, height)
    variance = losses.projected_variance(
        metric.means, metric.covariances, geometry.intrinsics_matrix(focal, width, height)
    )
    return grid, variance


def _headroom(variance, cap):
    median = float(variance.median())
    return {"median": median, "cap": cap, "sqrt_cap_over_median": math.sqrt(cap / median)}


def _splat_cap_check():
    depth = _room_depth(1536)
    rows = {}
    for init_size in (384, 1536):
        grid, variance = _projected_base(init_size, depth)
        _, scaled_cap = losses.splat_limits(grid)
        rows[str(init_size)] = {
            "grid": grid,
            "before": _headroom(variance, losses.SPLAT_MAX),
            "after": _headroom(variance, scaled_cap),
        }
    before_384 = rows["384"]["before"]["sqrt_cap_over_median"]
    before_1536 = rows["1536"]["before"]["sqrt_cap_over_median"]
    after_384 = rows["384"]["after"]["sqrt_cap_over_median"]
    rows["headroom_matches_1536"] = math.isclose(after_384, before_1536, rel_tol=0.1)
    rows["grid_192_was_tighter_than_2_5x"] = before_384 < 2.5
    rows["grid_768_headroom_above_6x"] = before_1536 > 6.0
    return rows


def _active_share(inverse, pixel_scale):
    grad = losses.gradient_magnitude(inverse, pixel_scale)
    share = float((grad > losses.FLOATER_EPS).float().mean())
    return {
        "share": share,
        "median_grad": float(grad.median()),
        "tv": float(losses.total_variation(inverse, pixel_scale)),
    }


def _floater_share_check():
    room = 1.0 / _room_depth(1536)
    steep = torch.linspace(0.05, 4.0, 1536).view(1, 1, 1, 1536).expand(1, 1, 1536, 1536)
    rows = {}
    for name, inverse in (("room", room), ("steep", steep)):
        coarse = torch.nn.functional.avg_pool2d(inverse, 4)
        rows[name] = {
            "1536_before": _active_share(inverse, 1.0),
            "384_before": _active_share(coarse, 1.0),
            "384_after": _active_share(coarse, losses.depth_pixel_scale(coarse)),
            "1536_after": _active_share(inverse, losses.depth_pixel_scale(inverse)),
        }
    steep = rows["steep"]
    room_row = rows["room"]
    rows["steep_384_before_exceeds_half"] = steep["384_before"]["share"] > 0.5
    rows["steep_1536_before_below_half"] = steep["1536_before"]["share"] < 0.5
    rows["steep_384_after_below_half"] = steep["384_after"]["share"] < 0.5
    rows["room_shares_below_half"] = (
        room_row["1536_before"]["share"] < 0.5 and room_row["384_before"]["share"] < 0.5
    )
    raw = losses.total_variation(torch.nn.functional.avg_pool2d(room, 4))
    scaled = losses.total_variation(
        torch.nn.functional.avg_pool2d(room, 4), 384 / losses.REFERENCE_DEPTH
    )
    rows["tv_scales_with_pixel"] = bool(torch.allclose(scaled, raw * 384 / losses.REFERENCE_DEPTH))
    return rows


def _synthetic_image(size=128):
    y = torch.linspace(0, 1, size).view(1, 1, size, 1)
    x = torch.linspace(0, 1, size).view(1, 1, 1, size)
    image = torch.cat(
        [
            y.expand(1, 1, size, size),
            x.expand(1, 1, size, size),
            torch.full((1, 1, size, size), 0.25),
        ],
        1,
    )
    image[:, :, 24:80, 36:100] = torch.tensor([0.85, 0.2, 0.15]).view(1, 3, 1, 1)
    image[:, :, 70:120, 60:118] = torch.tensor([0.15, 0.55, 0.8]).view(1, 3, 1, 1)
    return image.clamp(0, 1)


def _layer_parts(perceptual, rendered, target):
    with torch.no_grad():
        target_layers = perceptual.features(target)
        rendered_layers = perceptual.features(rendered)
        parts = []
        for rendered_layer, target_layer in zip(rendered_layers, target_layers, strict=True):
            feature, gram = losses.feature_and_gram(rendered_layer, target_layer)
            parts.append({"feature": float(feature), "gram": float(gram)})
    return parts


def _grad_norm(loss, leaf):
    leaf.grad = None
    loss.backward()
    return float(leaf.grad.norm())


def _perceptual_audit(perceptual):
    sharp = _synthetic_image()
    blurred = torch.nn.functional.avg_pool2d(sharp, 9, stride=1, padding=4)
    parts = _layer_parts(perceptual, blurred, sharp)
    feature = sum(part["feature"] for part in parts)
    gram = sum(part["gram"] for part in parts)
    rendered = blurred.detach().clone().requires_grad_(True)
    percep_grad = _grad_norm(perceptual(rendered, sharp), rendered)
    l1_grad = _grad_norm(losses.color_l1(rendered, sharp, torch.ones_like(sharp[:, :1])), rendered)
    ratio = percep_grad / l1_grad
    weighted = losses.STAGE3_WEIGHTS["percep"] * ratio
    recommend = losses.STAGE3_WEIGHTS["percep"] if 3.0 <= weighted <= 10.0 else 5.0 / ratio
    mask = torch.ones(1, 1, sharp.shape[-2], sharp.shape[-1])
    mask[:, :, :32, :] = 0
    black = blurred * mask
    composited = losses.in_frustum(blurred, sharp, mask)
    black_parts = _layer_parts(perceptual, black, sharp)
    composite_parts = _layer_parts(perceptual, composited, sharp)
    black_gram = sum(part["gram"] for part in black_parts)
    composite_gram = sum(part["gram"] for part in composite_parts)
    return {
        "layers": parts,
        "feature": feature,
        "gram": gram,
        "gram_over_feature": gram / feature,
        "percep_grad": percep_grad,
        "l1_grad": l1_grad,
        "percep_over_l1": ratio,
        "paper_weight_over_l1": weighted,
        "recommend": recommend,
        "paper_weight": losses.STAGE3_WEIGHTS["percep"],
        "black_gram": black_gram,
        "composite_gram": composite_gram,
        "black_leaks_more_than_composite": black_gram > composite_gram,
    }


def _dead_regularizer_check(perceptual):
    from sharp.models.alignment import LearnedAlignment
    from sharp.models.predictor import DepthAlignment

    from sharp_train.model import TrainingState, align_depth, hold_depth_alignment

    mono = torch.rand(1, 2, 32, 32) + 0.8
    gt = torch.rand(1, 1, 32, 32) + 0.8
    estimator = LearnedAlignment(steps=4, stride=1, base_width=16, depth_decoder_features=False)
    alignment = DepthAlignment(estimator)
    hold_depth_alignment(alignment)
    corrected, scale = align_depth(alignment, mono, gt, None, True)
    state = _state()
    state = TrainingState(
        aligned_depth=corrected,
        scale_map=scale,
        delta_xy=state.delta_xy,
        base_x=state.base_x,
        base_y=state.base_y,
    )
    intrinsics = torch.tensor([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]).view(
        1, 3, 3
    )
    batch = {
        "image_tgt": torch.rand(1, 3, 32, 32),
        "intrinsics": intrinsics,
        "mask_frustum_tgt": torch.ones(1, 1, 32, 32),
    }
    frozen = losses.stage3_terms(
        _view(batch["image_tgt"], torch.full((1, 1, 32, 32), 0.7)),
        _view(batch["image_tgt"], torch.full((1, 1, 32, 32), 0.7)),
        batch,
        _gaussians(),
        state,
        perceptual,
    )
    live = torch.nn.Parameter(torch.tensor(1.0))
    unused = torch.nn.Parameter(torch.tensor(1.0))
    total = live + frozen["tv"] + frozen["scale"] + frozen["grad_scale"]
    frozen_grads = torch.autograd.grad(total, [live, unused], allow_unused=True)
    depth = torch.rand(1, 2, 32, 32, requires_grad=True) + 0.5
    scale_map = torch.rand(1, 1, 32, 32, requires_grad=True) * 0.4 + 0.8
    decoder = torch.nn.Parameter(torch.zeros(3))
    open_state = TrainingState(
        aligned_depth=depth,
        scale_map=scale_map,
        delta_xy=state.delta_xy,
        base_x=state.base_x,
        base_y=state.base_y,
    )
    opened = losses.stage3_terms(
        _view(batch["image_tgt"], torch.full((1, 1, 32, 32), 0.7)),
        _view(batch["image_tgt"], torch.full((1, 1, 32, 32), 0.7)),
        batch,
        _gaussians(),
        open_state,
        perceptual,
    )
    open_grads = torch.autograd.grad(
        opened["tv"] + opened["scale"] + opened["grad_scale"],
        [depth, scale_map, decoder],
        allow_unused=True,
    )
    return {
        "held_depth_requires_grad": bool(corrected.requires_grad),
        "held_scale_requires_grad": bool(scale.requires_grad),
        "frozen_tv": float(frozen["tv"]),
        "frozen_scale": float(frozen["scale"]),
        "frozen_grad_scale": float(frozen["grad_scale"]),
        "frozen_live_connected": frozen_grads[0] is not None,
        "frozen_unused_is_none": frozen_grads[1] is None,
        "open_depth_grad": open_grads[0] is not None and float(open_grads[0].abs().sum()) > 0,
        "open_scale_grad": open_grads[1] is not None and float(open_grads[1].abs().sum()) > 0,
        "open_decoder_unused": open_grads[2] is None,
        "open_tv": float(opened["tv"].detach()),
        "open_scale": float(opened["scale"].detach()),
        "open_grad_scale": float(opened["grad_scale"].detach()),
    }


def _cli_defaults():
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "modal_app.py").read_text()
    warmup = int(re.search(r"^STAGE3_WARMUP = ([0-9_]+)", text, re.M).group(1).replace("_", ""))
    percep = [float(item) for item in re.findall(r"percep_weight: float = ([0-9.]+)", text)]
    return {
        "stage3_warmup": warmup,
        "percep_weight": percep,
        "novel_color_unmasked": "mask_frustum_tgt" not in text and "FULL_IMAGE" in text,
    }


def _alpha_hole_grad():
    empty = torch.zeros(1, 1, 8, 8, requires_grad=True)
    losses.alpha_bce(empty).backward()
    partial = torch.full((1, 1, 8, 8), 0.2, requires_grad=True)
    losses.alpha_bce(partial).backward()
    return {
        "empty_grad": float(empty.grad.abs().mean()),
        "partial_grad": float(partial.grad.abs().mean()),
    }


def cpu_checks():
    regularizers = _regularizer_checks()
    perceptual = losses.PerceptualLoss()
    percep_leaf = (torch.rand(1, 3, 32, 32) * 0.6 + 0.2).requires_grad_(True)
    target = torch.rand(1, 3, 32, 32) * 0.6 + 0.2
    regularizers["percep"] = _backward(perceptual(percep_leaf, target), percep_leaf)
    full_view = _full_view_check(perceptual)
    holes = _hole_fraction_check()
    depth = _depth_correction_check()
    scaling = _loss_scale_check()
    schedule = _schedule_check()
    alpha_target = _alpha_target_check(perceptual)
    hinges = _hinge_check()
    perceptual_scale = _perceptual_scale_check(perceptual)
    splat_cap = _splat_cap_check()
    floater_share = _floater_share_check()
    perceptual_audit = _perceptual_audit(perceptual)
    dead_terms = _dead_regularizer_check(perceptual)
    cli = _cli_defaults()
    hole_grad = _alpha_hole_grad()
    loss_names = ["alpha", "percep", "tv", "grad", "delta", "splat", "scale", "grad_scale"]
    criteria = {
        f"{name} finite with gradient": regularizers[name]["finite"]
        and regularizers[name]["grad"] > 0
        for name in loss_names
    }
    criteria["projected variance matches the screen-space formula"] = regularizers[
        "splat_identity"
    ]["matches"]
    criteria["hole fraction is the share of alpha < 0.5"] = bool(
        holes["matches"] and holes["frustum_masked_differs"]
    )
    criteria["novel-view losses train pixels outside the source frustum"] = bool(
        full_view["color_key"] == losses.FULL_IMAGE
        and full_view["color_changes_outside"]
        and full_view["alpha_changes_outside"]
        and full_view["percep_changes_outside"]
        and full_view["depth_ignores_outside"]
        and full_view["color_grad_outside"] > 0
        and full_view["alpha_grad_outside"] > 0
        and full_view["percep_grad_outside"] > 0
        and full_view["frustum_masked_color_grad_outside"] == 0
        and full_view["frustum_masked_alpha_grad_outside"] == 0
        and full_view["frustum_masked_percep_grad_outside"] == 0
        and full_view["perceptual_finite"]
        and full_view["wired_finite"]
        and full_view["surgery_matches_backward"]
        and full_view["frozen_depth_terms_are_zero"]
        and cli["novel_color_unmasked"]
    )
    criteria["depth correction changes depth only when enabled"] = bool(
        depth["init_is_identity"]
        and depth["enabled_changes_depth"]
        and depth["enabled_matches_exp_scale"]
        and depth["disabled_keeps_depth"]
        and depth["disabled_scale_is_one"]
        and depth["enabled_scale_away_from_one"]
    )
    criteria["scale regularizer overrides a ground-truth alignment"] = bool(
        depth["regularizer_increases_gt_error"]
    )
    criteria["held alignment leaves ground-truth depth unchanged"] = bool(
        depth["init_is_identity"]
        and depth["held_keeps_gt_depth"]
        and depth["base_depth_matches_gt"]
        and depth["base_depth_unchanged"]
        and depth["held_error_before"] < 1e-5
    )
    criteria["perceptual loss dominates at the paper weights"] = bool(
        scaling["paper_weights"]
        and scaling["early_percep_dominates"]
        and scaling["late_percep_dominates"]
    )
    criteria["global grad clip is 1"] = bool(
        scaling["clips_to_one"] and scaling["grad_clip"] == 1.0
    )
    criteria["paper warmup reaches 1.6e-4 at step 10000"] = bool(
        schedule["starts_at_peak_over_warmup"]
        and schedule["still_ramping_at_499"]
        and schedule["reaches_peak_at_end_of_warmup"]
        and schedule["cosine_continuous_at_warmup"]
        and schedule["cosine_ends_at_final"]
        and schedule["short_warmup_at_peak_by_499"]
        and schedule["probe_stays_on_ramp"]
    )
    criteria["short run warmup reaches the peak inside 4000 steps"] = bool(
        cli["stage3_warmup"] == losses.SHORT_RUN_WARMUP
        and schedule["short_run_reaches_peak"]
        and schedule["short_run_cosine_after_warmup"]
    )
    criteria["alpha bce targets 1 on the novel view"] = bool(
        alpha_target["wired_to_both_views"]
        and alpha_target["matches_bce_against_one"]
        and alpha_target["raising_novel_alpha_lowers_loss"]
    )
    criteria["splat loss above 0.1 is the upper hinge"] = bool(
        hinges["zero_variance_is_lower_cap"]
        and hinges["upper_hinge_is_048"]
        and hinges["inside_band_is_zero"]
        and hinges["offset_dead_zone"]
        and hinges["offset_fires_past_400"]
    )
    criteria["splat cap at grid 192 matches grid 768 headroom"] = bool(
        splat_cap["headroom_matches_1536"]
        and splat_cap["grid_192_was_tighter_than_2_5x"]
        and splat_cap["grid_768_headroom_above_6x"]
    )
    criteria["floater threshold tracks depth resolution"] = bool(
        floater_share["steep_384_before_exceeds_half"]
        and floater_share["steep_1536_before_below_half"]
        and floater_share["steep_384_after_below_half"]
        and floater_share["room_shares_below_half"]
        and floater_share["tv_scales_with_pixel"]
    )
    criteria["empty alpha keeps a gradient"] = bool(
        hole_grad["empty_grad"] > 0 and hole_grad["partial_grad"] > 0
    )
    criteria["perceptual audit is finite at the paper weight"] = bool(
        perceptual_audit["feature"] > 0
        and perceptual_audit["gram"] > 0
        and perceptual_audit["percep_grad"] > 0
        and perceptual_audit["l1_grad"] > 0
        and perceptual_audit["black_leaks_more_than_composite"]
        and perceptual_audit["paper_weight"] == 3.0
        and perceptual_audit["recommend"] > 0
        and cli["percep_weight"]
        and all(weight == 3.0 for weight in cli["percep_weight"])
    )
    criteria["held depth regularizers have no parameter gradient"] = bool(
        not dead_terms["held_depth_requires_grad"]
        and not dead_terms["held_scale_requires_grad"]
        and dead_terms["frozen_tv"] == 0.0
        and dead_terms["frozen_scale"] == 0.0
        and dead_terms["frozen_grad_scale"] == 0.0
        and dead_terms["frozen_live_connected"]
        and dead_terms["frozen_unused_is_none"]
        and dead_terms["open_depth_grad"]
        and dead_terms["open_scale_grad"]
        and dead_terms["open_decoder_unused"]
    )
    criteria["perceptual scaling matches appendix A.2"] = bool(
        perceptual_scale["feature_gram_matches_paper"]
        and perceptual_scale["identical_features_are_zero"]
        and perceptual_scale["imagenet_mean_maps_to_zero"]
        and perceptual_scale["half_uses_imagenet_stats"]
        and perceptual_scale["normalization_changes_features"]
    )
    return {
        "regularizers": regularizers,
        "full_view": full_view,
        "hole_fraction": holes,
        "depth_correction": depth,
        "loss_scale": scaling,
        "schedule": schedule,
        "alpha_target": alpha_target,
        "hinges": hinges,
        "perceptual_scale": perceptual_scale,
        "splat_cap": splat_cap,
        "floater_share": floater_share,
        "perceptual_audit": perceptual_audit,
        "dead_terms": dead_terms,
        "cli": cli,
        "hole_grad": hole_grad,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main():
    report = cpu_checks()
    printable = {
        "passed": report["passed"],
        "criteria": report["criteria"],
        "losses": {
            name: report["regularizers"][name]
            for name in ("alpha", "percep", "tv", "grad", "delta", "splat", "scale", "grad_scale")
        },
        "splat_identity": report["regularizers"]["splat_identity"],
        "full_view": report["full_view"],
        "hole_fraction": report["hole_fraction"],
        "depth_correction": report["depth_correction"],
        "loss_scale": report["loss_scale"],
        "schedule": report["schedule"],
        "alpha_target": report["alpha_target"],
        "hinges": report["hinges"],
        "perceptual_scale": report["perceptual_scale"],
        "splat_cap": report["splat_cap"],
        "floater_share": report["floater_share"],
        "perceptual_audit": report["perceptual_audit"],
        "dead_terms": report["dead_terms"],
        "cli": report["cli"],
        "hole_grad": report["hole_grad"],
    }
    print(json.dumps(printable, indent=1), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
