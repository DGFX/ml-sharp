import json
import math
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from sharp.models.gaussian_decoder import ImageFeatures
from sharp.models.initializer import create_initializer
from sharp.models.params import InitializerParams
from torch import nn

from sharp_train import losses, model, render, train, viz

FULL_GRID = 768


class PooledHead(nn.Module):
    def __init__(self, head, factor):
        super().__init__()
        self.head = head
        self.factor = factor

    def forward(self, features):
        return self.head(
            ImageFeatures(
                texture_features=F.avg_pool2d(features.texture_features, self.factor),
                geometry_features=F.avg_pool2d(features.geometry_features, self.factor),
            )
        )


def build_grid_predictor(checkpoint, grid, reset_decoder=True):
    predictor = model.build_predictor(checkpoint)
    if reset_decoder:
        model.reset_gaussian_decoder(predictor)
    factor = FULL_GRID // grid
    if factor > 1:
        params = InitializerParams()
        params.stride = 2 * factor
        predictor.init_model = create_initializer(params)
        predictor.prediction_head = PooledHead(predictor.prediction_head, factor)
    return predictor


def psnr(rendered, target, mask):
    mse = losses.masked_mean((rendered - target) ** 2, mask)
    return float(-10 * torch.log10(mse.clamp(min=1e-10)))


class PerceptualMetrics:
    def __init__(self, device):
        import lpips
        import piq

        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.dists = piq.DISTS().to(device).eval()

    def __call__(self, rendered, target):
        rendered = rendered.clamp(0, 1)
        return {
            "lpips": float(self.lpips(rendered * 2 - 1, target * 2 - 1).mean()),
            "dists": float(self.dists(rendered, target)),
        }


@torch.no_grad()
def evaluate(predictors, dataset, indices, device, out_dir, use_gt_depth=False, grid_pairs=4):
    metrics = PerceptualMetrics(device)
    out_dir = Path(out_dir)
    results = {name: [] for name in predictors}
    for position, index in enumerate(indices):
        batch = train.to_device(torch.utils.data.default_collate([dataset[index]]), device)
        _, _, height, width = batch["image_src"].shape
        renders = {}
        for name, predictor in predictors.items():
            predictor.eval()
            depth = batch["depth_src"] if use_gt_depth else None
            gaussians = model.predict(predictor, batch["image_src"], batch["focal_px"], depth)
            rendering = render.render(
                gaussians, batch["tgt_from_src"], batch["intrinsics"], width, height
            )
            renders[name] = rendering.color
            row = {"pair": batch["name"][0]}
            row.update(metrics(rendering.color, batch["image_tgt"]))
            row["psnr_visible"] = psnr(
                rendering.color, batch["image_tgt"], batch["mask_visible_tgt"]
            )
            row["l1_visible"] = float(
                losses.color_l1(rendering.color, batch["image_tgt"], batch["mask_visible_tgt"])
            )
            results[name].append(row)
        if position < grid_pairs:
            viz.save_rows(
                [
                    [viz.to_uint8(batch["image_src"][0]), viz.to_uint8(batch["image_tgt"][0])]
                    + [viz.to_uint8(color[0]) for color in renders.values()]
                ],
                out_dir / f"eval_{batch['name'][0]}.png",
            )
    summary = {}
    for name, rows in results.items():
        keys = [key for key in rows[0] if key != "pair"]
        summary[name] = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    return {
        "per_pair": results,
        "mean": summary,
        "order_in_grids": ["input", "target", *predictors],
    }


def look_at(eye, target):
    forward = target - eye
    forward = forward / forward.norm()
    down = torch.tensor([0.0, 1.0, 0.0], device=eye.device)
    right = torch.linalg.cross(down, forward)
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward], dim=0)
    viewmat = torch.eye(4, device=eye.device)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ eye
    return viewmat


@torch.no_grad()
def render_video(predictors, dataset, index, device, path, frames=90, radius_ratio=0.06):
    batch = train.to_device(torch.utils.data.default_collate([dataset[index]]), device)
    _, _, height, width = batch["image_src"].shape
    gaussians = {}
    for name, predictor in predictors.items():
        predictor.eval()
        gaussians[name] = model.predict(predictor, batch["image_src"], batch["focal_px"], None)
    first = next(iter(gaussians.values()))
    median_depth = float(first.means[0, ::17, 2].median())
    target = torch.tensor([0.0, 0.0, median_depth], device=device)
    radius = radius_ratio * median_depth
    video = []
    for step in range(frames):
        angle = 2 * math.pi * step / frames
        eye = torch.tensor(
            [
                radius * math.sin(angle),
                0.5 * radius * math.sin(2 * angle),
                radius * (1 - math.cos(angle)),
            ],
            device=device,
        )
        viewmat = look_at(eye, target)[None]
        row = []
        for value in gaussians.values():
            rendering = render.render(value, viewmat, batch["intrinsics"], width, height)
            row.append(viz.to_uint8(rendering.color[0]))
        video.append(np.concatenate(row, axis=1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=30, codec="libx264", quality=8, macro_block_size=8)
    for frame in video:
        writer.append_data(frame)
    writer.close()
    return {
        "pair": batch["name"][0],
        "frames": frames,
        "panels": list(predictors),
        "median_depth": median_depth,
    }


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))
