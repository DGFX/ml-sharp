import argparse
import json
import math
from pathlib import Path

import torch

from sharp_train import losses, model, multigrid, stage2, train
from sharp_train.data import HypersimPairs

CHECK_GRID = 192
GAUSSIAN_FIELDS = ("means", "covariances", "colors", "opacities")


class Stage3Loss:
    def __init__(self, device):
        self.perceptual = losses.PerceptualLoss().to(device)


def grad_norm(parameters):
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum(grad.square().sum() for grad in grads)))


def split_check(data_root):
    spec, val, train_scenes = stage2.held_out_split(data_root)
    held_out = HypersimPairs(data_root, names=spec["pairs"] + val["pairs"])
    return {
        "eval_scenes": spec["scenes"],
        "validation_scene": val["scene"],
        "train_scenes": train_scenes,
        "validation_pairs": len(val["pairs"]),
        "loaded_pairs": len(held_out),
        "criteria": {
            "validation scene is not an eval scene": val["scene"] not in spec["scenes"],
            "validation scene is not a train scene": val["scene"] not in train_scenes,
            "eval scenes are not train scenes": not set(spec["scenes"]) & set(train_scenes),
            "validation pairs share no pair with eval": not set(val["pairs"]) & set(spec["pairs"]),
            "validation pairs come from the validation scene": all(
                name.startswith(val["scene"] + "_") for name in val["pairs"]
            ),
            "eval and validation pairs load from the data root": len(held_out)
            == len(spec["pairs"]) + len(val["pairs"]),
        },
    }


def train_step_regularizers(predictor, batch, depth_correction):
    predictor.zero_grad(set_to_none=True)
    gaussians = model.predict(
        predictor,
        batch["image_src"],
        batch["focal_px"],
        batch["depth_src"],
        depth_correction=depth_correction,
    )
    state = predictor.training_state
    terms = losses.regularizer_terms(batch, gaussians, state)
    total = sum(losses.STAGE3_WEIGHTS[key] * value for key, value in terms.items())
    total.backward()
    alignment = list(predictor.predictor.depth_alignment.parameters())
    decoder = list(predictor.predictor.prediction_head.parameters())
    return gaussians, {
        "scale_minus_one": float((state.scale_map.detach() - 1).abs().mean()),
        "terms": {key: float(value.detach()) for key, value in terms.items()},
        "total_finite": math.isfinite(float(total.detach())),
        "alignment_grad": grad_norm(alignment),
        "decoder_grad": grad_norm(decoder),
        "alignment_trainable": all(parameter.requires_grad for parameter in alignment),
    }


def depth_path_check(predictor, batch):
    trainable = {id(parameter) for parameter in model.trainable_parameters(predictor)}
    alignment = list(predictor.predictor.depth_alignment.parameters())
    predictor.eval()
    with torch.no_grad():
        inference = model.predict(predictor, batch["image_src"], batch["focal_px"], None)
    predictor.train()
    default, default_report = train_step_regularizers(predictor, batch, False)
    unfrozen, unfrozen_report = train_step_regularizers(predictor, batch, True)
    predictor.zero_grad(set_to_none=True)
    default_matches = {
        field: bool(torch.equal(getattr(default, field).detach(), getattr(inference, field)))
        for field in GAUSSIAN_FIELDS
    }
    depth_gap = (unfrozen.means[..., 2].detach() - inference.means[..., 2]).abs().mean()
    return {
        "default_train_matches_eval_inference": default_matches,
        "default_train": default_report,
        "unfrozen_train": unfrozen_report,
        "unfrozen_vs_inference_mean_abs_z": float(depth_gap),
        "inference_median_z": float(inference.means[..., 2].median()),
        "criteria": {
            "default training Gaussians equal the eval inference Gaussians": all(
                default_matches.values()
            ),
            "default training scale map is 1": default_report["scale_minus_one"] == 0.0,
            "default training leaves the alignment without gradient": default_report[
                "alignment_grad"
            ]
            == 0.0,
            "default training still trains the decoder": default_report["decoder_grad"] > 0,
            "alignment is in the optimizer": all(
                id(parameter) in trainable for parameter in alignment
            ),
            "--depth-correction 1 trains the alignment": unfrozen_report["alignment_trainable"]
            and unfrozen_report["alignment_grad"] > 0
            and math.isfinite(unfrozen_report["alignment_grad"]),
            "regularizers are finite": default_report["total_finite"]
            and unfrozen_report["total_finite"],
        },
    }


def spike_attribution_check(predictor, batch):
    predictor.train()
    predictor.zero_grad(set_to_none=True)
    gaussians = model.predict(
        predictor, batch["image_src"], batch["focal_px"], batch["depth_src"], depth_correction=True
    )
    terms = losses.regularizer_terms(batch, gaussians, predictor.training_state)
    weighted = losses.weighted_terms(terms)
    terms["total"] = sum(weighted.values())
    parameters = model.trainable_parameters(predictor)
    norms = train.term_grad_norms(terms, parameters)
    terms["total"].backward()
    optimizer_norm = float(torch.nn.utils.clip_grad_norm_(parameters, math.inf))
    predictor.zero_grad(set_to_none=True)
    term_norms = {key: value for key, value in norms.items() if key != "total"}
    live = {key for key, value in weighted.items() if value.requires_grad}
    return {
        "term_grad_norms": norms,
        "optimizer_grad_norm": optimizer_norm,
        "criteria": {
            "spike total norm equals the optimizer gradient norm": abs(
                norms["total"] - optimizer_norm
            )
            <= 1e-4 * optimizer_norm,
            "every live weighted term gets a finite norm": set(term_norms) == live
            and all(math.isfinite(value) for value in norms.values()),
            "term norms bound the total norm": norms["total"]
            <= sum(term_norms.values()) * (1 + 1e-5),
        },
    }


def perceptual_check(batch):
    perceptual = losses.PerceptualLoss()
    rendered, target = batch["image_src"], batch["image_tgt"]
    direct = rendered.clone().requires_grad_(True)
    perceptual.value(direct, target).backward()
    surgery = rendered.clone().requires_grad_(True)
    value = perceptual(surgery, target)
    value.backward()
    value = value.detach()
    same = perceptual(target.clone(), target)
    difference = (direct.grad - surgery.grad).abs().max() / direct.grad.abs().max()
    return {
        "value": float(value),
        "value_on_target": float(same),
        "grad_relative_max_difference": float(difference),
        "criteria": {
            "eager perceptual gradient equals full backward": float(difference) < 1e-3,
            "perceptual is zero on the target": float(same) < 1e-6,
            "perceptual is positive on the source view": float(value) > 1e-6,
        },
    }


def cpu_checks(data_root, apple_path):
    split = split_check(data_root)
    val = stage2.val_spec()
    dataset = HypersimPairs(data_root, names=val["pairs"][:1])
    batch = torch.utils.data.default_collate([dataset[0]])
    torch.manual_seed(0)
    predictor = stage2.build_grid_predictor(str(apple_path), CHECK_GRID, reset_decoder=False)
    depth = depth_path_check(predictor, batch)
    spikes = spike_attribution_check(predictor, batch)
    del predictor
    perceptual = perceptual_check(batch)
    criteria = {
        **split["criteria"],
        **depth["criteria"],
        **spikes["criteria"],
        **perceptual["criteria"],
    }
    return {
        "pair": batch["name"][0],
        "grid": CHECK_GRID,
        "split": split,
        "depth_path": depth,
        "spike_attribution": spikes,
        "perceptual": perceptual,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    report = cpu_checks(args.data, multigrid.ensure_apple())
    print(json.dumps(report, indent=1), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
