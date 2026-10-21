import argparse
import itertools
import json
import math
import tempfile
from pathlib import Path

import torch

from sharp_train import losses, model, multigrid, stage2, train
from sharp_train.data import EpochShuffleSampler, HypersimPairs, train_stream

CHECK_GRID = 192
RESUME_STEPS = 4
RESUME_SPLIT = 2
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


def regularizer_forward(predictor, batch, mask_key=None, stage3=None, depth_correction=True):
    gaussians = model.predict(
        predictor,
        batch["image_src"],
        batch["focal_px"],
        batch["depth_src"],
        depth_correction=depth_correction,
    )
    terms = losses.regularizer_terms(batch, gaussians, predictor.training_state)
    terms["total"] = sum(losses.weighted_terms(terms).values())
    return terms, None, None


def spike_attribution_check(predictor, batch):
    predictor.train()
    predictor.zero_grad(set_to_none=True)
    terms, _, _ = regularizer_forward(predictor, batch)
    weighted = losses.weighted_terms(terms)
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


def train_segment(apple_path, checkpoint, dataset, out_dir, start=0, resume=None, stop_after=None):
    predictor, _ = stage2.load_checkpoint(str(apple_path), checkpoint)
    captured = {}

    def callback(step, last, state):
        stopping = step + 1 == stop_after
        if stopping:
            stage2.save_checkpoint(
                predictor, Path(out_dir) / "resume.pt", step + 1, {"training": state()}
            )
        if stopping or last:
            captured.update(state())
        return stopping

    train.train(
        predictor,
        train_stream(dataset, 0, start),
        RESUME_STEPS,
        out_dir,
        lr=1e-4,
        warmup=2,
        log_every=3,
        callback=callback,
        resume=resume,
        forward=regularizer_forward,
    )
    trainable = {
        name
        for name, parameter in predictor.predictor.named_parameters()
        if parameter.requires_grad
    }
    return stage2.trained_state(predictor), captured, trainable


def max_difference(first, second):
    return max(float((first[key].float() - second[key].float()).abs().max()) for key in first)


def optimizer_difference(first, second):
    return max(
        max_difference(first["state"][index], second["state"][index]) for index in first["state"]
    )


def without_seconds(history):
    return [{key: value for key, value in entry.items() if key != "seconds"} for entry in history]


def resume_check(data_root, apple_path):
    val = stage2.val_spec()
    dataset = HypersimPairs(data_root, names=val["pairs"])
    size = len(dataset)
    stream = list(itertools.islice(EpochShuffleSampler(size, 0), 2 * size))
    seeked = list(itertools.islice(EpochShuffleSampler(size, 0, size - 1), size + 1))
    workdir = Path(tempfile.mkdtemp())
    initial = workdir / "initial.pt"
    torch.manual_seed(0)
    source = stage2.build_grid_predictor(str(apple_path), CHECK_GRID, reset_decoder=False)
    stage2.save_checkpoint(source, initial, 0)
    del source
    straight, straight_end, trainable = train_segment(
        apple_path, initial, dataset, workdir / "straight"
    )
    _, stopped, _ = train_segment(
        apple_path, initial, dataset, workdir / "stopped", stop_after=RESUME_SPLIT
    )
    saved = workdir / "stopped" / "resume.pt"
    payload = stage2.read_resume(saved)
    resumed, resumed_end, _ = train_segment(
        apple_path,
        saved,
        dataset,
        workdir / "resumed",
        start=payload["training"]["next_step"],
        resume=payload["training"],
    )
    restarted, _, _ = train_segment(
        apple_path,
        saved,
        dataset,
        workdir / "restarted",
        start=RESUME_SPLIT,
        stop_after=RESUME_STEPS - RESUME_SPLIT,
    )
    weight_gap = max_difference(straight, resumed)
    optimizer_gap = optimizer_difference(straight_end["optimizer"], resumed_end["optimizer"])
    restart_gap = max_difference(straight, restarted)
    histories_match = without_seconds(straight_end["history"]) == without_seconds(
        resumed_end["history"]
    )
    return {
        "steps": RESUME_STEPS,
        "split": RESUME_SPLIT,
        "saved_next_step": payload["training"]["next_step"],
        "saved_window": len(payload["training"]["window_norms"]),
        "weight_max_difference": weight_gap,
        "optimizer_max_difference": optimizer_gap,
        "restart_without_state_weight_difference": restart_gap,
        "learning_rates": [entry["lr"] for entry in resumed_end["history"]],
        "criteria": {
            "seeked sampler continues the stream across an epoch": seeked
            == stream[size - 1 : 2 * size],
            "checkpoint resumes at the next step mid log window": stopped["next_step"]
            == payload["training"]["next_step"]
            == RESUME_SPLIT
            and len(payload["training"]["window_norms"]) == 1,
            "checkpoint holds every trainable parameter": trainable <= set(payload["state_dict"]),
            "resumed weights equal uninterrupted weights": weight_gap == 0.0,
            "resumed Adam state equals uninterrupted Adam state": optimizer_gap == 0.0
            and straight_end["optimizer"]["param_groups"][0]["lr"]
            == resumed_end["optimizer"]["param_groups"][0]["lr"],
            "resumed history and learning rates equal uninterrupted": histories_match,
            "resumed RNG state equals uninterrupted": torch.equal(
                straight_end["rng"]["cpu"], resumed_end["rng"]["cpu"]
            ),
            "restart without training state diverges": restart_gap > 0.0,
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
    resume = resume_check(data_root, apple_path)
    criteria = {
        **split["criteria"],
        **depth["criteria"],
        **spikes["criteria"],
        **perceptual["criteria"],
        **resume["criteria"],
    }
    return {
        "pair": batch["name"][0],
        "grid": CHECK_GRID,
        "split": split,
        "depth_path": depth,
        "spike_attribution": spikes,
        "perceptual": perceptual,
        "resume": resume,
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
