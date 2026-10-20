from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image


def to_uint8(image):
    array = image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (array * 255 + 0.5).astype(np.uint8)


def colorize_disparity(depth, valid=None, max_disparity=None):
    disparity = 1.0 / torch.nan_to_num(depth, nan=1e6).clamp(min=1e-3)
    if max_disparity is None:
        max_disparity = float(torch.quantile(disparity.flatten()[::97].float(), 0.99))
    normalized = (disparity / max(max_disparity, 1e-6)).clamp(0, 1)[0].detach().cpu().numpy()
    colored = matplotlib.colormaps["turbo"](normalized)[..., :3]
    if valid is not None:
        colored = colored * valid[0].detach().cpu().numpy()[..., None]
    return (colored * 255 + 0.5).astype(np.uint8)


def save_rows(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = np.concatenate([np.concatenate(row, axis=1) for row in rows], axis=0)
    Image.fromarray(grid).save(path)


def save_pair_grid(batch, rendering_src, rendering_tgt, path, index=0):
    max_disparity = float(
        torch.quantile(
            (1.0 / torch.nan_to_num(batch["depth_src"][index], nan=1e6).clamp(min=1e-3)).flatten()[
                ::97
            ],
            0.99,
        )
    )
    rows = [
        [
            to_uint8(batch["image_src"][index]),
            to_uint8(rendering_src.color[index]),
            colorize_disparity(batch["depth_src"][index], max_disparity=max_disparity),
            colorize_disparity(rendering_src.depth[index], max_disparity=max_disparity),
        ],
        [
            to_uint8(batch["image_tgt"][index]),
            to_uint8(rendering_tgt.color[index]),
            colorize_disparity(batch["depth_tgt"][index], max_disparity=max_disparity),
            colorize_disparity(rendering_tgt.depth[index], max_disparity=max_disparity),
        ],
    ]
    save_rows(rows, path)


def save_curve(history, keys, path, title):
    figure = Figure(figsize=(7, 4))
    axis = figure.add_subplot()
    steps = [entry["step"] for entry in history]
    for key in keys:
        axis.plot(steps, [entry[key] for entry in history], label=key)
    axis.set_yscale("log")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
