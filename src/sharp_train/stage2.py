import json
import math
from pathlib import Path

import torch

from sharp_train import losses, model

EVAL_PAIRS_PATH = Path(__file__).with_name("eval_pairs.json")
VAL_PAIRS_PATH = Path(__file__).with_name("val_pairs.json")


def eval_spec():
    return json.loads(EVAL_PAIRS_PATH.read_text())


def val_spec():
    return json.loads(VAL_PAIRS_PATH.read_text())


def held_out_split(data_root, train_scenes=""):
    spec, val = eval_spec(), val_spec()
    held_out = {*spec["scenes"], val["scene"]}
    pairs = json.loads((Path(data_root) / "pairs.json").read_text())
    scenes = train_scenes.split(",") if train_scenes else sorted({pair["scene"] for pair in pairs})
    return spec, val, [scene for scene in scenes if scene not in held_out]


def build_grid_predictor(
    checkpoint, grid, reset_decoder=True, grad_checkpointing=False, backbone_bf16=False
):
    predictor = model.build_predictor(checkpoint, grad_checkpointing=grad_checkpointing)
    if reset_decoder:
        model.reset_gaussian_decoder(predictor)
    return model.GridPredictor(predictor, grid, backbone_bf16=backbone_bf16)


def trained_state(grid_predictor):
    return {
        name: value.detach().cpu()
        for name, value in grid_predictor.predictor.state_dict().items()
        if not name.startswith("monodepth_model.")
    }


def save_checkpoint(grid_predictor, path, step, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": grid_predictor.grid,
        "step": step,
        "state_dict": trained_state(grid_predictor),
    }
    payload.update(extra or {})
    partial = path.with_suffix(".partial")
    torch.save(payload, partial)
    partial.rename(path)


def load_checkpoint(
    apple_checkpoint, path, grid=None, grad_checkpointing=False, backbone_bf16=False
):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source_grid = payload["grid"]
    target_grid = source_grid if grid is None else grid
    grid_predictor = build_grid_predictor(
        apple_checkpoint,
        target_grid,
        reset_decoder=False,
        grad_checkpointing=grad_checkpointing,
        backbone_bf16=backbone_bf16,
    )
    result = grid_predictor.predictor.load_state_dict(payload["state_dict"], strict=False)
    unexpected_missing = [
        name for name in result.missing_keys if not name.startswith("monodepth_model.")
    ]
    if unexpected_missing or result.unexpected_keys:
        raise KeyError(f"checkpoint mismatch: {unexpected_missing} {result.unexpected_keys}")
    model.rescale_base_scale(grid_predictor, source_grid)
    return grid_predictor, payload["step"]


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

    def region_lpips(self, rendered, target, region):
        region = region.expand_as(rendered)
        composited = rendered * region + target * (1.0 - region)
        return float(self.lpips(composited.clamp(0, 1) * 2 - 1, target.clamp(0, 1) * 2 - 1).mean())


def pair_metrics(metrics, rendering_src, rendering_tgt, batch):
    visible = batch["mask_visible_tgt"]
    valid_src = losses.valid_depth_mask(batch["depth_src"])
    valid_tgt = losses.valid_depth_mask(batch["depth_tgt"]) * visible
    occluded = 1.0 - visible
    row = metrics(rendering_tgt.color, batch["image_tgt"])
    row.update(
        {
            "lpips_occluded": metrics.region_lpips(
                rendering_tgt.color, batch["image_tgt"], occluded
            ),
            "hole_fraction": float(losses.hole_fraction(rendering_tgt.alpha)),
            "hole_fraction_frustum": float(
                losses.hole_fraction(rendering_tgt.alpha, mask=batch["mask_frustum_tgt"])
            ),
            "psnr_tgt_visible": psnr(rendering_tgt.color, batch["image_tgt"], visible),
            "color_l1_src": float(
                losses.color_l1(rendering_src.color, batch["image_src"], torch.ones_like(visible))
            ),
            "color_l1_tgt_visible": float(
                losses.color_l1(rendering_tgt.color, batch["image_tgt"], visible)
            ),
            "disparity_l1_src": float(
                losses.disparity_l1(rendering_src.depth, batch["depth_src"], valid_src)
            ),
            "disparity_l1_tgt_visible": float(
                losses.disparity_l1(rendering_tgt.depth, batch["depth_tgt"], valid_tgt)
            ),
            "visible_fraction": float(visible.mean()),
        }
    )
    return row


def comparison_rows(batch, renders):
    from sharp_train import viz

    max_disparity = float(
        torch.quantile(
            (1.0 / torch.nan_to_num(batch["depth_src"][0], nan=1e6).clamp(min=1e-3)).flatten()[
                ::97
            ],
            0.99,
        )
    )
    mask = batch["mask_visible_tgt"][0].expand(3, -1, -1)
    colors = [viz.to_uint8(batch["image_src"][0]), viz.to_uint8(batch["image_tgt"][0])]
    disparities = [
        viz.to_uint8(mask),
        viz.colorize_disparity(batch["depth_tgt"][0], max_disparity=max_disparity),
    ]
    for rendering in renders.values():
        colors.append(viz.to_uint8(rendering.color[0]))
        disparities.append(viz.colorize_disparity(rendering.depth[0], max_disparity=max_disparity))
    return [colors, disparities]


@torch.no_grad()
def evaluate(predictors, dataset, device, out_dir, settings=("inference", "gt_depth"), images=6):
    from sharp_train import render, train, viz

    metrics = PerceptualMetrics(device)
    out_dir = Path(out_dir)
    results = {setting: {name: [] for name in predictors} for setting in settings}
    for index in range(len(dataset)):
        batch = train.to_device(torch.utils.data.default_collate([dataset[index]]), device)
        _, _, height, width = batch["image_src"].shape
        for setting in settings:
            depth = batch["depth_src"] if setting == "gt_depth" else None
            renders = {}
            for name, predictor in predictors.items():
                predictor.eval()
                gaussians = model.predict(predictor, batch["image_src"], batch["focal_px"], depth)
                rendering_src, rendering_tgt = render.render_pair(gaussians, batch, width, height)
                renders[name] = rendering_tgt
                row = {"pair": batch["name"][0], "gaussians": int(gaussians.means.shape[1])}
                row.update(pair_metrics(metrics, rendering_src, rendering_tgt, batch))
                results[setting][name].append(row)
            if setting == "inference" and index < images:
                viz.save_rows(
                    comparison_rows(batch, renders), out_dir / f"compare_{batch['name'][0]}.png"
                )
        print(json.dumps({"pair": batch["name"][0], "done": index + 1}), flush=True)
    summary = {}
    for setting, by_model in results.items():
        summary[setting] = {}
        for name, rows in by_model.items():
            keys = [key for key in rows[0] if key not in ("pair", "gaussians")]
            summary[setting][name] = {
                key: sum(row[key] for row in rows) / len(rows) for key in keys
            }
            summary[setting][name]["gaussians"] = rows[0]["gaussians"]
    return {
        "pairs": [row["pair"] for row in next(iter(results[settings[0]].values()))],
        "per_pair": results,
        "mean": summary,
        "image_layout": {
            "row_0": ["input", "target", *predictors],
            "row_1": ["visibility mask", "target disparity", *predictors],
        },
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


def camera_path(median_depth, frames, radius_ratio, device):
    target = torch.tensor([0.0, 0.0, median_depth], device=device)
    radius = radius_ratio * median_depth
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
        yield look_at(eye, target)[None]


def write_video(frames, path, fps=30):
    import imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def downscale(frame, factor):
    height = frame.shape[0] // factor // 8 * 8
    width = frame.shape[1] // factor // 8 * 8
    tensor = torch.from_numpy(frame).permute(2, 0, 1)[None].float()
    small = torch.nn.functional.interpolate(tensor, size=(height, width), mode="area")
    return small[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


@torch.no_grad()
def render_videos(predictors, dataset, index, device, out_dir, frames=90, radius_ratio=0.06):
    import numpy as np

    from sharp_train import render, train, viz

    batch = train.to_device(torch.utils.data.default_collate([dataset[index]]), device)
    _, _, height, width = batch["image_src"].shape
    name = batch["name"][0]
    gaussians = {}
    for label, predictor in predictors.items():
        predictor.eval()
        gaussians[label] = model.predict(predictor, batch["image_src"], batch["focal_px"], None)
    reference = next(iter(gaussians.values()))
    median_depth = float(reference.means[0, ::17, 2].median())
    panels = {label: [] for label in gaussians}
    for viewmat in camera_path(median_depth, frames, radius_ratio, device):
        for label, value in gaussians.items():
            rendering = render.render(value, viewmat, batch["intrinsics"], width, height)
            panels[label].append(viz.to_uint8(rendering.color[0]))
    out_dir = Path(out_dir)
    paths = {}
    for label, video in panels.items():
        paths[label] = str(out_dir / f"video_{name}_{label}.mp4")
        write_video(video, paths[label])
    side_by_side = [np.concatenate(row, axis=1) for row in zip(*panels.values())]
    paths["side_by_side"] = str(out_dir / f"video_{name}_side_by_side.mp4")
    write_video(side_by_side, paths["side_by_side"])
    paths["side_by_side_small"] = str(out_dir / f"video_{name}_side_by_side_small.mp4")
    write_video(
        [downscale(frame, 2 * len(panels)) for frame in side_by_side], paths["side_by_side_small"]
    )
    return {
        "pair": name,
        "frames": frames,
        "fps": 30,
        "panels": list(panels),
        "median_depth": median_depth,
        "files": paths,
    }


@torch.no_grad()
def validation_loss(predictor, batches, device, stage3=None, depth_correction=True):
    from sharp_train import train

    was_training = predictor.training
    predictor.eval()
    totals = {}
    for batch in batches:
        terms, _, _ = train.forward_pair(
            predictor,
            train.to_device(batch, device),
            mask_key="mask_visible_tgt",
            stage3=stage3,
            depth_correction=depth_correction,
        )
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value) / len(batches)
    predictor.train(was_training)
    return totals


def validation_batches(data_root, names):
    from sharp_train.data import HypersimPairs

    dataset = HypersimPairs(data_root, names=names)
    return [torch.utils.data.default_collate([dataset[index]]) for index in range(len(dataset))]


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))
