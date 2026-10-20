import json
import math

import torch

from sharp_train import losses


class Stage3Loss:
    def __init__(self, device):
        self.perceptual = losses.PerceptualLoss().to(device)


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
    return {"hole_fraction": value, "expected": 0.5, "matches": math.isclose(value, 0.5)}


def _depth_correction_check():
    from sharp.models.alignment import LearnedAlignment
    from sharp.models.predictor import DepthAlignment

    from sharp_train.model import align_depth

    estimator = LearnedAlignment(steps=4, stride=1, base_width=16, depth_decoder_features=False)
    alignment = DepthAlignment(estimator)
    monodepth = torch.rand(1, 2, 32, 32) + 0.5
    gt_depth = torch.rand(1, 1, 32, 32) + 0.5
    initial, _ = align_depth(alignment, monodepth, gt_depth, None, True)
    estimator.conv_out.bias.data.fill_(1.0)
    enabled, scale = align_depth(alignment, monodepth, gt_depth, None, True)
    disabled, disabled_scale = align_depth(alignment, monodepth, gt_depth, None, False)
    return {
        "init_is_identity": bool(torch.allclose(initial, monodepth, atol=1e-5)),
        "enabled_matches_exp_scale": bool(
            torch.allclose(enabled, monodepth * math.e, rtol=1e-4, atol=1e-4)
        ),
        "enabled_changes_depth": bool((enabled - monodepth).abs().mean() > 1e-3),
        "disabled_keeps_depth": bool(torch.equal(disabled, monodepth)),
        "disabled_scale_is_one": bool(torch.equal(disabled_scale, torch.ones_like(disabled_scale))),
        "enabled_scale_away_from_one": bool((scale - 1).abs().mean() > 1e-3),
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


def _full_view_check(perceptual):
    torch.manual_seed(0)
    color = torch.rand(1, 3, 32, 32) * 0.6 + 0.2
    target = torch.rand(1, 3, 32, 32) * 0.6 + 0.2
    alpha = torch.full((1, 1, 32, 32), 0.7)
    mask = torch.ones(1, 1, 32, 32)
    mask[:, :, :, :8] = 0
    changed_color = color.clone()
    changed_color[:, :, :, :4] = 0.05
    changed_alpha = alpha.clone()
    changed_alpha[:, :, :, :4] = 0.05
    state = _state()
    gaussians = _gaussians()
    intrinsics = torch.tensor([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]).view(
        1, 3, 3
    )
    batch = {"image_tgt": target, "intrinsics": intrinsics, "mask_visible_tgt": mask}
    with torch.no_grad():
        original = losses.stage3_terms(
            _view(color, alpha), _view(color, alpha), batch, gaussians, state, perceptual
        )
        changed = losses.stage3_terms(
            _view(color, alpha),
            _view(changed_color, changed_alpha),
            batch,
            gaussians,
            state,
            perceptual,
        )
    masked = losses.color_l1(color, target, mask)
    masked_changed = losses.color_l1(changed_color, target, mask)

    color_leaf = changed_color.detach().clone().requires_grad_(True)
    alpha_leaf = changed_alpha.detach().clone().requires_grad_(True)
    wired = losses.stage3_terms(
        _view(color, alpha), _view(color_leaf, alpha_leaf), batch, gaussians, state, perceptual
    )
    (wired["alpha"] + wired["percep"]).backward()

    direct = color.detach().clone().requires_grad_(True)
    perceptual.value(direct, target).backward()
    through_surgery = color.detach().clone().requires_grad_(True)
    perceptual(through_surgery, target).backward()
    return {
        "masked_color_ignores_occluded_pixels": bool(torch.allclose(masked, masked_changed)),
        "perceptual_responds_outside_mask": bool(
            not torch.allclose(original["percep"], changed["percep"])
        ),
        "alpha_responds_outside_mask": bool(
            not torch.allclose(original["alpha"], changed["alpha"])
        ),
        "perceptual_grad_outside_mask": float(color_leaf.grad[:, :, :, :8].abs().sum()),
        "alpha_grad_outside_mask": float(alpha_leaf.grad[:, :, :, :8].abs().sum()),
        "surgery_matches_backward": bool(
            torch.allclose(direct.grad, through_surgery.grad, rtol=1e-3, atol=1e-4)
        ),
        "perceptual_finite": bool(torch.isfinite(original["percep"])),
        "wired_finite": all(bool(torch.isfinite(value).all()) for value in original.values()),
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
    loss_names = ["alpha", "percep", "tv", "grad", "delta", "splat", "scale", "grad_scale"]
    criteria = {
        f"{name} finite with gradient": regularizers[name]["finite"]
        and regularizers[name]["grad"] > 0
        for name in loss_names
    }
    criteria["projected variance matches the screen-space formula"] = regularizers[
        "splat_identity"
    ]["matches"]
    criteria["hole fraction is the share of alpha < 0.5"] = holes["matches"]
    criteria["full-view losses respond outside the visibility mask"] = bool(
        full_view["masked_color_ignores_occluded_pixels"]
        and full_view["perceptual_responds_outside_mask"]
        and full_view["alpha_responds_outside_mask"]
        and full_view["perceptual_grad_outside_mask"] > 0
        and full_view["alpha_grad_outside_mask"] > 0
        and full_view["perceptual_finite"]
        and full_view["wired_finite"]
        and full_view["surgery_matches_backward"]
    )
    criteria["depth correction changes depth only when enabled"] = bool(
        depth["init_is_identity"]
        and depth["enabled_changes_depth"]
        and depth["enabled_matches_exp_scale"]
        and depth["disabled_keeps_depth"]
        and depth["disabled_scale_is_one"]
        and depth["enabled_scale_away_from_one"]
    )
    return {
        "regularizers": regularizers,
        "full_view": full_view,
        "hole_fraction": holes,
        "depth_correction": depth,
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
    }
    print(json.dumps(printable, indent=1), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
