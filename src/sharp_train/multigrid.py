import gc
import json
import tempfile
import urllib.request
from pathlib import Path

import torch

from sharp_train import model, stage2, train

CACHE = Path("/tmp/sharp-ckpts") / model.APPLE_CHECKPOINT_NAME
SOURCE_GRID = 192
GRID = 384


def ensure_apple():
    if CACHE.exists() and CACHE.stat().st_size > 1_000_000_000:
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    partial = CACHE.with_suffix(".partial")
    urllib.request.urlretrieve(model.APPLE_CHECKPOINT_URL, partial)
    partial.rename(CACHE)
    return CACHE


def set_checkpointing(grid_predictor, enabled):
    feature = grid_predictor.predictor.feature_model
    feature.grad_checkpointing = enabled
    feature.decoder.set_grad_checkpointing(enabled)


def decoder_backward(grid_predictor):
    predictor = grid_predictor.predictor
    predictor.train()
    predictor.zero_grad(set_to_none=True)
    dims = predictor.monodepth_model.get_feature_dims()
    levels = len(dims)
    coarse = 2
    sizes = [coarse * 2 ** (levels - 1 - index) for index in range(levels)]
    encodings = [torch.randn(1, dim, size, size) for dim, size in zip(dims, sizes)]
    incoming = torch.randn(1, predictor.feature_model.dim_in, sizes[0] * 2, sizes[0] * 2)
    delta = predictor.prediction_head(predictor.feature_model(incoming, encodings))
    delta.sum().backward()
    return {
        "delta": list(delta.shape),
        "dtype": str(delta.dtype).removeprefix("torch."),
        "feature_grad": train.grad_norm(predictor.feature_model.parameters()),
        "head_grad": train.grad_norm(predictor.prediction_head.parameters()),
        "backbone_grad": train.grad_norm(predictor.monodepth_model.parameters()),
        "grad_checkpointing": bool(predictor.feature_model.grad_checkpointing)
        and bool(predictor.feature_model.decoder.grad_checkpointing),
    }


def backbone_probe(grid_predictor, enabled):
    encoder = grid_predictor.predictor.monodepth_model.monodepth_predictor.encoder.image_encoder
    block = encoder.blocks[0]
    frozen = [parameter.requires_grad for parameter in block.parameters()]
    block.requires_grad_(True)
    grid_predictor.backbone_bf16 = enabled
    grid_predictor.zero_grad(set_to_none=True)
    image = torch.rand(1, 3, encoder.internal_resolution(), encoder.internal_resolution())
    output = model.apply_backbone(grid_predictor, encoder, image)
    features = output[0]
    if features.requires_grad:
        features.sum().backward()
    probe = {
        "enabled": enabled,
        "dtype": str(features.dtype).removeprefix("torch."),
        "requires_grad": bool(features.requires_grad),
        "block_grad": train.grad_norm(block.parameters()),
    }
    for parameter, flag in zip(block.parameters(), frozen):
        parameter.requires_grad_(flag)
    grid_predictor.backbone_bf16 = False
    grid_predictor.zero_grad(set_to_none=True)
    return probe


def count_gaussians(grid_predictor):
    grid_predictor.to("meta")
    image = torch.empty(1, 3, model.INTERNAL_RESOLUTION, model.INTERNAL_RESOLUTION, device="meta")
    factor = torch.empty(1, device="meta")
    with torch.no_grad():
        gaussians = grid_predictor(image, factor, None, depth_correction=False)
    return int(gaussians.mean_vectors.shape[1])


def cpu_checks(apple_path):
    source = stage2.build_grid_predictor(str(apple_path), SOURCE_GRID, reset_decoder=False)
    name, weight = next(
        (key, value.detach().clone())
        for key, value in source.predictor.feature_model.state_dict().items()
        if value.ndim == 4
    )
    checkpoint = Path(tempfile.mkstemp(suffix=".pt")[1])
    stage2.save_checkpoint(source, checkpoint, 8000)
    del source
    gc.collect()
    loaded, step = stage2.load_checkpoint(
        str(apple_path), checkpoint, grid=GRID, grad_checkpointing=True
    )
    copied = loaded.predictor.feature_model.state_dict()[name]
    initializer = loaded.predictor.init_model
    scale_192 = float(model.median_base_scale(initializer, model.resolution_for_grid(SOURCE_GRID)))
    scale_384 = float(model.median_base_scale(initializer, loaded.input_resolution))
    scale_768 = float(
        model.median_base_scale(initializer, model.resolution_for_grid(model.FULL_GRID))
    )
    base = initializer(
        torch.zeros(1, 3, loaded.input_resolution, loaded.input_resolution),
        torch.ones(1, 2, loaded.input_resolution, loaded.input_resolution),
    ).gaussian_base_values
    layers = int(base.mean_inverse_z_ndc.shape[2])
    per_layer = int(base.mean_x_ndc.shape[-2] * base.mean_x_ndc.shape[-1])
    with_checkpointing = decoder_backward(loaded)
    set_checkpointing(loaded, False)
    without_checkpointing = decoder_backward(loaded)
    bf16 = backbone_probe(loaded, True)
    fp32 = backbone_probe(loaded, False)
    gaussians = count_gaussians(loaded)
    checkpoint.unlink(missing_ok=True)
    ratio_768 = scale_768 / scale_192
    criteria = {
        "checkpoint step is the saved step": step == 8000,
        "decoder conv weights survive a grid change": bool(torch.equal(copied, weight)),
        "grid is 384": loaded.grid == GRID,
        "scale prior ratio is source/target": abs(loaded.scale_prior_ratio - SOURCE_GRID / GRID)
        < 1e-6,
        "192 to 384 base scale is 1/2": abs(scale_384 / scale_192 - 0.5) < 1e-3,
        "192 to 768 base scale is 1/4": abs(ratio_768 - 0.25) < 1e-3,
        "two layers": layers == 2,
        "gaussians are both layers": gaussians == layers * per_layer == 2 * GRID * GRID,
        "grad checkpointing off has decoder grads and a frozen backbone": bool(
            not without_checkpointing["grad_checkpointing"]
            and without_checkpointing["feature_grad"] > 0
            and without_checkpointing["head_grad"] > 0
            and without_checkpointing["backbone_grad"] == 0
            and without_checkpointing["dtype"] == "float32"
        ),
        "grad checkpointing on has decoder grads and a frozen backbone": bool(
            with_checkpointing["grad_checkpointing"]
            and with_checkpointing["feature_grad"] > 0
            and with_checkpointing["head_grad"] > 0
            and with_checkpointing["backbone_grad"] == 0
            and with_checkpointing["dtype"] == "float32"
        ),
        "backbone bf16 has no backbone grads and fp32 output": bool(
            bf16["enabled"]
            and bf16["dtype"] == "float32"
            and not bf16["requires_grad"]
            and bf16["block_grad"] == 0
        ),
        "backbone fp32 still receives grads when unfrozen": bool(
            not fp32["enabled"] and fp32["requires_grad"] and fp32["block_grad"] > 0
        ),
    }
    return {
        "grid": GRID,
        "source_grid": SOURCE_GRID,
        "layers": layers,
        "per_layer": per_layer,
        "gaussians": gaussians,
        "expected": 2 * GRID * GRID,
        "scale_192": scale_192,
        "scale_384": scale_384,
        "scale_768": scale_768,
        "scale_prior_ratio": loaded.scale_prior_ratio,
        "without_grad_checkpointing": without_checkpointing,
        "with_grad_checkpointing": with_checkpointing,
        "backbone_bf16": bf16,
        "backbone_fp32": fp32,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main():
    report = cpu_checks(ensure_apple())
    print(json.dumps(report, indent=1), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
