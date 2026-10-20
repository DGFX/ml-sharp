import json
import math
import re
import time
from pathlib import Path

import torch
from sharp.utils.gaussians import compose_covariance_matrices, unproject_gaussians

from sharp_train import geometry, losses, model, render, train, viz

ATTRIBUTE_CHANNELS = {
    "mean_xy": slice(0, 2),
    "mean_z": slice(2, 3),
    "scale": slice(3, 6),
    "quaternion": slice(6, 10),
    "color": slice(10, 13),
    "opacity": slice(13, 14),
}
FORWARD_PATH = [
    "models",
    "utils/gaussians.py",
    "utils/gsplat.py",
    "utils/linalg.py",
    "utils/math.py",
    "utils/color_space.py",
]
INFERENCE_PATTERN = re.compile(
    r"no_grad|inference_mode|\.eval\(\)|\.detach\(\)|requires_grad_\(False\)|\.numpy\(\)"
)


def scan_inference_only_code(sharp_root):
    findings = []
    for entry in FORWARD_PATH:
        path = Path(sharp_root) / entry
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            for number, line in enumerate(file.read_text().splitlines(), start=1):
                if INFERENCE_PATTERN.search(line):
                    findings.append(f"{file.relative_to(sharp_root)}:{number}: {line.strip()}")
    return findings


def attribute_grad_norms(captured):
    gradient = captured["delta"].grad
    return {
        name: float(gradient[:, channels].abs().mean())
        for name, channels in ATTRIBUTE_CHANNELS.items()
    }


def apple_unprojection_forward(predictor, batch):
    _, _, height, width = batch["image_src"].shape
    captured = {}

    def hook(output):
        output.retain_grad()
        captured["delta"] = output

    resized = torch.nn.functional.interpolate(
        batch["image_src"],
        size=(model.INTERNAL_RESOLUTION,) * 2,
        mode="bilinear",
        align_corners=True,
    )
    depth = torch.nn.functional.interpolate(
        model.fill_invalid_depth(batch["depth_src"]),
        size=(model.INTERNAL_RESOLUTION,) * 2,
        mode="nearest",
    )
    handle = predictor.prediction_head.register_forward_hook(lambda _, __, output: hook(output))
    gaussians_ndc = predictor(resized, batch["focal_px"] / width, depth)
    handle.remove()
    intrinsics = torch.linalg.inv(model.ndc_to_camera_matrix(batch["focal_px"], width, height))
    ndc = torch.tensor(
        [
            [2.0 / model.INTERNAL_RESOLUTION, 0, -1, 0],
            [0, 2.0 / model.INTERNAL_RESOLUTION, -1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=intrinsics.device,
    )
    intrinsics = torch.linalg.inv(ndc) @ intrinsics[0]
    gaussians = unproject_gaussians(
        gaussians_ndc,
        torch.eye(4, device=intrinsics.device),
        intrinsics,
        (model.INTERNAL_RESOLUTION,) * 2,
    )
    metric = model.MetricGaussians(
        gaussians.mean_vectors,
        compose_covariance_matrices(gaussians.quaternions, gaussians.singular_values),
        gaussians.colors,
        gaussians.opacities,
    )
    rendering_src, rendering_tgt = render.render_pair(metric, batch, width, height)
    terms = losses.pair_losses(rendering_src, rendering_tgt, batch)
    terms["total"].backward()
    return attribute_grad_norms(captured)


def gradient_check(predictor, batch, sharp_root):
    predictor.train()
    predictor.zero_grad(set_to_none=True)
    captured = {}

    def hook(output):
        output.retain_grad()
        captured["delta"] = output

    terms, _, _ = train.forward_pair(predictor, batch, delta_hook=hook)
    terms["total"].backward()
    ours = attribute_grad_norms(captured)
    core = predictor.predictor if isinstance(predictor, model.GridPredictor) else predictor
    groups = {
        "feature_model": core.feature_model,
        "prediction_head": core.prediction_head,
        "depth_alignment": core.depth_alignment,
    }
    group_norms = {}
    for name, module in groups.items():
        named = [(key, p) for key, p in module.named_parameters() if p.requires_grad]
        with_grad = [p.grad for _, p in named if p.grad is not None]
        group_norms[name] = {
            "trainable_tensors": len(named),
            "tensors_with_grad": len(with_grad),
            "without_grad": [key for key, p in named if p.grad is None],
            "grad_norm": float(torch.sqrt(sum((g.float() ** 2).sum() for g in with_grad)))
            if with_grad
            else 0.0,
            "finite": all(bool(torch.isfinite(g).all()) for g in with_grad),
        }
    monodepth_trainable = sum(
        p.numel() for p in core.monodepth_model.parameters() if p.requires_grad
    )
    predictor.zero_grad(set_to_none=True)
    apple = None
    if core is predictor:
        apple = apple_unprojection_forward(predictor, batch)
        predictor.zero_grad(set_to_none=True)
    result = {
        "grad_enabled": torch.is_grad_enabled(),
        "static_scan": scan_inference_only_code(sharp_root),
        "delta_grad_ours": ours,
        "delta_grad_apple_unproject": apple,
        "module_grads": group_norms,
        "monodepth_trainable_params": monodepth_trainable,
    }
    result["passed"] = bool(
        result["grad_enabled"]
        and all(value > 0 for value in ours.values())
        and all(
            group["tensors_with_grad"] >= 0.95 * group["trainable_tensors"] and group["finite"]
            for group in group_norms.values()
        )
        and group_norms["feature_model"]["grad_norm"] > 0
    )
    return result


@torch.no_grad()
def apple_weights_check(predictor, dataset, indices, out_dir, device):
    predictor.eval()
    out_dir = Path(out_dir)
    rows = []
    for index in indices:
        batch = train.to_device(torch.utils.data.default_collate([dataset[index]]), device)
        _, _, height, width = batch["image_src"].shape
        gaussians = model.predict(
            predictor, batch["image_src"], batch["focal_px"], batch["depth_src"]
        )
        rendering_src, rendering_tgt = render.render_pair(gaussians, batch, width, height)
        full = losses.pair_losses(rendering_src, rendering_tgt, batch)
        masked = losses.pair_losses(
            rendering_src, rendering_tgt, batch, tgt_mask_key="mask_frustum_tgt"
        )
        wrong_batch = dict(batch, tgt_from_src=torch.linalg.inv(batch["tgt_from_src"]))
        _, wrong_tgt = render.render_pair(gaussians, wrong_batch, width, height)
        wrong = losses.color_l1(wrong_tgt.color, batch["image_tgt"], batch["mask_frustum_tgt"])
        identity = losses.color_l1(
            batch["image_src"], batch["image_tgt"], batch["mask_frustum_tgt"]
        )
        mono = model.predict(predictor, batch["image_src"], batch["focal_px"], None)
        mono_src, mono_tgt = render.render_pair(mono, batch, width, height)
        mono_terms = losses.pair_losses(mono_src, mono_tgt, batch, tgt_mask_key="mask_frustum_tgt")
        name = batch["name"][0]
        viz.save_pair_grid(batch, rendering_src, rendering_tgt, out_dir / f"apple_{name}.png")
        row = {
            "pair": name,
            "color_src": float(full["color_src"]),
            "color_tgt": float(full["color_tgt"]),
            "color_tgt_frustum": float(masked["color_tgt"]),
            "depth_src": float(full["depth_src"]),
            "depth_tgt_frustum": float(masked["depth_tgt"]),
            "total": float(full["total"]),
            "control_wrong_pose_color_tgt": float(wrong),
            "control_identity_color_tgt": float(identity),
            "control_no_gt_depth_color_tgt": float(mono_terms["color_tgt"]),
            "alpha_src_mean": float(rendering_src.alpha.mean()),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    keys = [key for key in rows[0] if key != "pair"]
    mean = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    criteria = {
        "color_src < 0.05": mean["color_src"] < 0.05,
        "color_tgt_frustum < 0.10": mean["color_tgt_frustum"] < 0.10,
        "color_tgt_frustum < 0.6 * identity": mean["color_tgt_frustum"]
        < 0.6 * mean["control_identity_color_tgt"],
        "every pair: correct pose beats inverted pose": all(
            row["color_tgt_frustum"] < row["control_wrong_pose_color_tgt"] for row in rows
        ),
        "depth_src disparity L1 < 0.05": mean["depth_src"] < 0.05,
        "alpha_src_mean > 0.95": mean["alpha_src_mean"] > 0.95,
    }
    return {"pairs": rows, "mean": mean, "criteria": criteria, "passed": all(criteria.values())}


def overfit_check(predictor, dataset, index, out_dir, device, steps, lr):
    out_dir = Path(out_dir)
    model.reset_gaussian_decoder(predictor)
    predictor.to(device)
    batch = torch.utils.data.default_collate([dataset[index]])
    device_batch = train.to_device(batch, device)
    before = train.snapshot(
        predictor, device_batch, out_dir / "overfit_step_000000.png", "mask_frustum_tgt"
    )
    history = train.train(
        predictor,
        [batch],
        steps,
        out_dir,
        lr=lr,
        warmup=20,
        mask_key="mask_frustum_tgt",
        log_every=10,
        snapshot_every=max(steps // 4, 1),
        snapshot_batch=device_batch,
    )
    after = train.snapshot(
        predictor, device_batch, out_dir / "overfit_final.png", "mask_frustum_tgt"
    )
    viz.save_curve(
        history,
        ["total", "color_src", "color_tgt", "depth_src", "depth_tgt"],
        out_dir / "overfit_curve.png",
        f"overfit {batch['name'][0]}",
    )
    criteria = {
        "total drops below 50% of initial": after["total"] < 0.5 * before["total"],
        "color_src < 0.02": after["color_src"] < 0.02,
        "color_tgt_frustum < 0.04": after["color_tgt"] < 0.04,
    }
    return {
        "pair": batch["name"][0],
        "before": before,
        "after": after,
        "history": history,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def reset_decoder_check(predictor, batch):
    head = predictor.prediction_head
    zero_head = all(float(parameter.abs().max()) == 0.0 for parameter in head.parameters())
    predictor.train()
    predictor.zero_grad(set_to_none=True)
    terms, _, _ = train.forward_pair(predictor, batch, mask_key="mask_visible_tgt")
    terms["total"].backward()
    head_grad = float(
        torch.sqrt(
            sum((p.grad.float() ** 2).sum() for p in head.parameters() if p.grad is not None)
        )
    )
    predictor.zero_grad(set_to_none=True)
    result = {
        "head_starts_at_zero": zero_head,
        "head_grad_norm": head_grad,
        "terms_at_zero_delta": train.summarize(terms),
    }
    result["passed"] = bool(zero_head and head_grad > 0 and math.isfinite(head_grad))
    return result


@torch.no_grad()
def grid_equivalence_check(predictor, batch):
    predictor.eval()
    _, _, _, width = batch["image_src"].shape
    resized = torch.nn.functional.interpolate(
        batch["image_src"],
        size=(model.INTERNAL_RESOLUTION,) * 2,
        mode="bilinear",
        align_corners=True,
    )
    depth = torch.nn.functional.interpolate(
        model.fill_invalid_depth(batch["depth_src"]),
        size=(model.INTERNAL_RESOLUTION,) * 2,
        mode="nearest",
    )
    disparity_factor = batch["focal_px"] / width
    apple = predictor(resized, disparity_factor, depth)
    wrapped = model.GridPredictor(predictor, model.FULL_GRID)(resized, disparity_factor, depth)
    differences = {
        "mean_vectors": float((apple.mean_vectors - wrapped.mean_vectors).abs().max()),
        "singular_values": float((apple.singular_values - wrapped.singular_values).abs().max()),
        "colors": float((apple.colors - wrapped.colors).abs().max()),
        "opacities": float((apple.opacities - wrapped.opacities).abs().max()),
    }
    return {
        "gaussians": int(apple.mean_vectors.shape[1]),
        "max_abs_difference": differences,
        "passed": all(value == 0.0 for value in differences.values()),
    }


@torch.no_grad()
def grid_forward_check(predictor, batch, grid):
    grid_predictor = model.GridPredictor(predictor, grid).eval()
    _, _, height, width = batch["image_src"].shape
    gaussians = model.predict(
        grid_predictor, batch["image_src"], batch["focal_px"], batch["depth_src"]
    )
    rendering_src, rendering_tgt = render.render_pair(gaussians, batch, width, height)
    terms = losses.pair_losses(rendering_src, rendering_tgt, batch, tgt_mask_key="mask_visible_tgt")
    result = {
        "grid": grid,
        "input_resolution": grid_predictor.input_resolution,
        "gaussians": int(gaussians.means.shape[1]),
        "apple_decoder_terms": train.summarize(terms),
        "alpha_src_mean": float(rendering_src.alpha.mean()),
    }
    result["passed"] = bool(
        result["gaussians"] == 2 * grid * grid
        and result["apple_decoder_terms"]["color_src"] < 0.1
        and result["alpha_src_mean"] > 0.9
    )
    return result


def visibility_check(dataset, indices):
    rows = []
    for index in indices:
        sample = dataset[index]
        visible = sample["mask_visible_tgt"][None]
        frustum = sample["mask_frustum_tgt"][None]
        src_from_tgt = torch.linalg.inv(sample["tgt_from_src"])[None]
        warped = geometry.warp_src_to_tgt(
            sample["image_src"][None],
            sample["depth_tgt"][None],
            sample["intrinsics"][None],
            src_from_tgt,
        )
        error = (warped - sample["image_tgt"][None]).abs().mean(dim=1, keepdim=True)
        strict_frustum = geometry.frustum_mask(
            sample["depth_tgt"][None], sample["intrinsics"][None], src_from_tgt, margin=1.0
        )
        occluded = strict_frustum * (1 - visible)
        rows.append(
            {
                "pair": sample["name"],
                "visible_fraction": float(visible.mean()),
                "frustum_fraction": float(frustum.mean()),
                "occluded_fraction": float(occluded.mean()),
                "visible_outside_frustum": float((visible * (1 - frustum)).sum()),
                "warp_error_visible": float(losses.masked_mean(error, visible)),
                "warp_error_occluded": float(losses.masked_mean(error, occluded)),
            }
        )
    with_occlusion = [row for row in rows if row["occluded_fraction"] > 0.01]
    criteria = {
        "visible is inside the frustum": all(row["visible_outside_frustum"] == 0 for row in rows),
        "visible is a strict subset on some pairs": len(with_occlusion) > 0,
        "every pair: warp error visible < occluded": all(
            row["warp_error_visible"] < row["warp_error_occluded"] for row in with_occlusion
        ),
        "mean warp error visible < 0.5 x occluded": mean_of(rows, "warp_error_visible")
        < 0.5 * mean_of(with_occlusion, "warp_error_occluded"),
        "every pair: warp error visible < 0.1": all(
            row["warp_error_visible"] < 0.1 for row in rows
        ),
    }
    return {
        "pairs": rows,
        "mean": {key: mean_of(rows, key) for key in rows[0] if key != "pair"},
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def mean_of(rows, key):
    return sum(row[key] for row in rows) / max(len(rows), 1)


def throughput_check(predictor, loader, steps, out_dir):
    torch.cuda.reset_peak_memory_stats()
    started = None
    timings = []

    def callback(step, last):
        nonlocal started
        torch.cuda.synchronize()
        now = time.time()
        if started is not None:
            timings.append(now - started)
        started = now

    history = train.train(
        predictor,
        loader,
        steps,
        out_dir,
        lr=1e-4,
        warmup=5,
        mask_key="mask_visible_tgt",
        log_every=5,
        callback=callback,
    )
    steady = timings[len(timings) // 2 :]
    return {
        "steps": steps,
        "seconds_per_step": sum(steady) / len(steady),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "first_total": history[0]["total"],
        "last_total": history[-1]["total"],
        "finite": all(math.isfinite(entry["total"]) for entry in history),
    }
