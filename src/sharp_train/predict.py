import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch
from sharp.cli import predict as apple_predict
from sharp.utils import io
from sharp.utils.gaussians import compose_covariance_matrices, load_ply, save_ply

from sharp_train import model, multigrid, stage2

CHECK_GRID = 384
MEANS_REL_TOL = 1e-3
COVARIANCE_REL_TOL = 1e-3
ATTRIBUTE_ABS_TOL = 1e-4


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_ply(apple_checkpoint, checkpoint, image_path, out_path, device="cpu", grid=None):
    grid_predictor, step = stage2.load_checkpoint(
        str(apple_checkpoint), checkpoint, grid=grid or None
    )
    grid_predictor = grid_predictor.to(device).eval()
    image, _, f_px = io.load_rgb(Path(image_path))
    height, width = image.shape[:2]
    gaussians = apple_predict.predict_image(grid_predictor, image, f_px, torch.device(device))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(gaussians, f_px, (height, width), out_path)
    return {
        "gaussians": int(gaussians.mean_vectors.shape[1]),
        "grid": grid_predictor.grid,
        "step": step,
        "f_px": float(f_px),
        "image_size": [width, height],
        "out": str(out_path),
    }


def reference_gaussians(grid_predictor, image, f_px):
    tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1)[None] / 255.0
    with torch.no_grad():
        return model.predict(grid_predictor.eval(), tensor, torch.tensor([float(f_px)]))


def relative_error(value, reference, dims):
    difference = (value - reference).flatten(dims).norm(dim=-1)
    return difference / reference.flatten(dims).norm(dim=-1)


def cpu_check(apple_checkpoint, image_path, workdir, grid=CHECK_GRID):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint = workdir / f"apple_grid{grid}.pt"
    out_path = workdir / "check.ply"
    image, _, f_px = io.load_rgb(Path(image_path))
    height, width = image.shape[:2]
    source = stage2.build_grid_predictor(str(apple_checkpoint), grid, reset_decoder=False)
    stage2.save_checkpoint(source, checkpoint, step=0)
    reference = reference_gaussians(source, image, f_px)
    del source
    gc.collect()
    summary = predict_ply(apple_checkpoint, checkpoint, image_path, out_path, "cpu")
    ply, metadata = load_ply(out_path)

    means_rel = relative_error(ply.mean_vectors, reference.means, 2)
    ply_covariances = compose_covariance_matrices(ply.quaternions, ply.singular_values)
    covariance_rel = relative_error(ply_covariances, reference.covariances, 2)
    colors_abs = (ply.colors - reference.colors).abs()
    opacities_abs = (ply.opacities - reference.opacities).abs()
    expected = 2 * grid * grid
    attributes = (
        ply.mean_vectors,
        ply.singular_values,
        ply.quaternions,
        ply.colors,
        ply.opacities,
    )
    criteria = {
        "gaussian count is 2 * grid^2": summary["gaussians"] == expected
        and ply.mean_vectors.shape[1] == expected,
        "summary grid is the checkpoint grid": summary["grid"] == grid,
        "all attributes are finite": all(bool(torch.isfinite(item).all()) for item in attributes),
        "opacities are in (0, 1)": bool(((ply.opacities > 0) & (ply.opacities < 1)).all()),
        "scales are positive": bool((ply.singular_values > 0).all()),
        "ply means match the model means": float(means_rel.median()) < MEANS_REL_TOL,
        "ply covariances match the model covariances": float(covariance_rel.median())
        < COVARIANCE_REL_TOL,
        "ply opacities match the model opacities": float(opacities_abs.max()) < ATTRIBUTE_ABS_TOL,
        "ply colors match the model colors": float(colors_abs.max()) < ATTRIBUTE_ABS_TOL,
        "metadata has the image focal length and size": abs(metadata.focal_length_px - f_px)
        < 1e-3 * f_px
        and tuple(int(size) for size in metadata.resolution_px) == (width, height),
        "metadata colors are linearRGB after load": metadata.color_space == "linearRGB",
    }
    return {
        "image": str(image_path),
        "grid": grid,
        "expected_gaussians": expected,
        "summary": summary,
        "means": {
            "max_abs": float((ply.mean_vectors - reference.means).abs().max()),
            "median_relative": float(means_rel.median()),
            "max_relative": float(means_rel.max()),
        },
        "covariances": {
            "median_relative": float(covariance_rel.median()),
            "max_relative": float(covariance_rel.max()),
            "compared_as": "ply quaternions and scales composed into covariances",
        },
        "colors": {
            "max_abs": float(colors_abs.max()),
            "compared_as": "model linearRGB vs ply sRGB SH DC converted back by load_ply",
        },
        "opacities": {
            "max_abs": float(opacities_abs.max()),
            "compared_as": "model opacity against sigmoid of the stored logit",
        },
        "tolerances": {
            "means_median_relative": MEANS_REL_TOL,
            "covariances_median_relative": COVARIANCE_REL_TOL,
            "attribute_max_abs": ATTRIBUTE_ABS_TOL,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict a .ply Gaussian splat from one image.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--grid", type=int)
    parser.add_argument("--apple", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    apple = args.apple or multigrid.ensure_apple()
    if args.check:
        with tempfile.TemporaryDirectory() as workdir:
            report = cpu_check(apple, args.image, workdir, args.grid or CHECK_GRID)
        print(json.dumps(report, indent=1), flush=True)
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --check is given")
    out = args.out or Path(args.image.stem + ".ply")
    summary = predict_ply(apple, args.checkpoint, args.image, out, args.device, args.grid)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
