import json
import os
from pathlib import Path

import modal

APP_NAME = "ml-sharp-train"
VOLUME_NAME = "ml-sharp-data"
VOLUME_ROOT = Path("/vol")
DATA_ROOT = VOLUME_ROOT / "hypersim"
RUNS_ROOT = VOLUME_ROOT / "runs"
GPU = "A100-40GB"
H100 = "H100"
BASE_GRID = 192
BASE_MAX_MINUTES = 50.0
STAGE3_TIMEOUT_MINUTES = {"check": 45, "train": 120, "eval": 60}
STAGE2_TIMEOUT_MINUTES = {"checks": 25, "train": 80, "eval": 40}
FIXED_TIMEOUT_MINUTES = {"stage1": 45, "predict": 15}
TRAIN_TASKS = ("stage2-train", "stage3-train")
GPU_USD_PER_HOUR = {"A100-40GB": 2.23, "H100": 4.20}
MODAL_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
STARTUP_MINUTES = 10
TIMEOUT_MARGIN_MINUTES = 15

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "build-essential", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.14.1", "torchvision==0.29.1", index_url="https://download.pytorch.org/whl/cu126"
    )
    .pip_install("numpy", "ninja", "jaxtyping", "rich", "setuptools", "wheel")
    .run_commands(
        "pip download --no-deps --no-binary gsplat gsplat==1.5.3 -d /tmp/gsplat",
        "tar xzf /tmp/gsplat/gsplat-1.5.3.tar.gz -C /tmp/gsplat",
        "sed -i 's/-std=c++17/-std=c++20/' /tmp/gsplat/gsplat-1.5.3/setup.py",
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9' MAX_JOBS=16 "
        "pip install --no-build-isolation /tmp/gsplat/gsplat-1.5.3",
        "rm -rf /tmp/gsplat",
    )
    .pip_install(
        "timm==1.0.30",
        "click",
        "imageio[ffmpeg]==2.38.1",
        "matplotlib",
        "pillow-heif",
        "plyfile",
        "scipy",
        "pillow",
        "lpips==0.1.4",
        "piq==0.8.0",
    )
    .run_commands(
        "python -c 'from torchvision.models import ResNet50_Weights, resnet50; "
        "resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)'",
        "python -c 'import lpips, piq; lpips.LPIPS(net=\"alex\", verbose=False); piq.DISTS()'",
    )
    .env({"PYTHONPATH": "/root/repo/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("src", "/root/repo/src")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def ensure_checkpoint():
    import urllib.request

    from sharp_train.model import APPLE_CHECKPOINT_NAME, APPLE_CHECKPOINT_URL

    path = VOLUME_ROOT / "checkpoints" / APPLE_CHECKPOINT_NAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        urllib.request.urlretrieve(APPLE_CHECKPOINT_URL, partial)
        partial.rename(path)
        volume.commit()
    return path


def cuda_setup():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def volume_path(path):
    path = Path(path)
    return path if path.is_absolute() else VOLUME_ROOT / path


STAGE3_INIT = "runs/stage3-b29749e/checkpoints/final.pt"


def budget_scale(grid):
    return (grid / BASE_GRID) ** 2


def resolved_max_minutes(max_minutes, grid):
    if max_minutes:
        return max_minutes
    return BASE_MAX_MINUTES * budget_scale(grid)


def resolved_timeout_seconds(timeout_minutes, base_minutes, grid):
    minutes = timeout_minutes if timeout_minutes else base_minutes * budget_scale(grid)
    return int(minutes * 60)


def resolved_gpu(gpu, grid):
    if gpu:
        return gpu
    if grid >= 768:
        return H100
    return GPU


def run_options(base_minutes, grid, gpu, timeout_minutes):
    return {
        "gpu": resolved_gpu(gpu, grid),
        "timeout": resolved_timeout_seconds(timeout_minutes, base_minutes, grid),
    }


def task_options(task, mode, grid, gpu, timeout_minutes):
    if task in FIXED_TIMEOUT_MINUTES:
        return {"gpu": GPU, "timeout": FIXED_TIMEOUT_MINUTES[task] * 60}
    if task == "stage3" and mode in STAGE3_TIMEOUT_MINUTES:
        gpu = gpu or os.environ.get("STAGE3_GPU", "")
        return run_options(STAGE3_TIMEOUT_MINUTES[mode], grid, gpu, timeout_minutes)
    kind = task.removeprefix("stage2-")
    if kind != task and kind in STAGE2_TIMEOUT_MINUTES:
        return run_options(STAGE2_TIMEOUT_MINUTES[kind], grid, gpu, timeout_minutes)
    raise SystemExit(f"unknown --task {task} --mode {mode}")


def preflight(task, gpu, timeout_seconds, max_minutes, max_usd, seconds_per_step, steps):
    if gpu not in GPU_USD_PER_HOUR:
        raise SystemExit(f"refused: no price for GPU {gpu}")
    price = GPU_USD_PER_HOUR[gpu]
    timeout_minutes = timeout_seconds / 60
    report = {
        "task": task,
        "gpu": gpu,
        "usd_per_hour": price,
        "timeout_minutes": timeout_minutes,
        "worst_case_usd": round(price * timeout_minutes / 60, 2),
        "max_usd": max_usd,
    }
    training = task in TRAIN_TASKS
    if training:
        report["budget_minutes"] = max_minutes
    if training and seconds_per_step:
        train_minutes = min(steps * seconds_per_step / 60, max_minutes)
        report["expected_usd"] = round(price * (train_minutes + STARTUP_MINUTES) / 60, 2)
    reasons = []
    if not max_usd:
        reasons.append("pass --max-usd")
    elif report["worst_case_usd"] > max_usd:
        reasons.append("worst_case_usd exceeds --max-usd")
    if timeout_seconds > MODAL_MAX_TIMEOUT_SECONDS:
        reasons.append("timeout exceeds Modal's 24 h limit")
    if training and max_minutes + TIMEOUT_MARGIN_MINUTES > timeout_minutes:
        reasons.append(f"max_minutes + {TIMEOUT_MARGIN_MINUTES} exceeds the timeout")
    if reasons:
        raise SystemExit(f"refused ({'; '.join(reasons)}): {json.dumps(report)}")
    return report


@app.function(gpu=GPU, timeout=45 * 60, volumes={VOLUME_ROOT: volume})
def stage1(
    run_name: str, eval_scenes: str, num_apple_pairs: int, overfit_steps: int, overfit_lr: float
):
    import torch
    from sharp_train import checks, model, stage2
    from sharp_train.data import HypersimPairs

    device = cuda_setup()
    out_dir = RUNS_ROOT / run_name
    checkpoint = ensure_checkpoint()
    scenes = eval_scenes.split(",")
    dataset = HypersimPairs(DATA_ROOT, scenes=scenes)
    stride = max(len(dataset) // num_apple_pairs, 1)
    indices = list(range(0, len(dataset), stride))[:num_apple_pairs]
    predictor = model.build_predictor(checkpoint).to(device)
    report = {"gpu": torch.cuda.get_device_name(), "pairs_available": len(dataset)}
    batch = torch.utils.data.default_collate([dataset[indices[0]]])
    from sharp_train.train import to_device

    report["gradient_check"] = checks.gradient_check(
        predictor, to_device(batch, device), "/root/repo/src/sharp"
    )
    print(
        json.dumps(
            {key: value for key, value in report["gradient_check"].items() if key != "static_scan"}
        ),
        flush=True,
    )
    report["apple_weights"] = checks.apple_weights_check(
        predictor, dataset, indices, out_dir, device
    )
    print(
        json.dumps(
            {
                "apple_mean": report["apple_weights"]["mean"],
                "criteria": report["apple_weights"]["criteria"],
            }
        ),
        flush=True,
    )
    stage2.save_json(out_dir / "stage1_report.json", report)
    volume.commit()
    report["overfit"] = checks.overfit_check(
        predictor, dataset, indices[0], out_dir, device, overfit_steps, overfit_lr
    )
    report["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    report["passed"] = bool(
        report["gradient_check"]["passed"]
        and report["apple_weights"]["passed"]
        and report["overfit"]["passed"]
    )
    stage2.save_json(out_dir / "stage1_report.json", report)
    volume.commit()
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "overfit_criteria": report["overfit"]["criteria"],
                "after": report["overfit"]["after"],
            }
        ),
        flush=True,
    )
    return {"passed": report["passed"], "out_dir": str(out_dir)}


def train_loader(scenes, seed, start=0):
    from sharp_train.data import HypersimPairs, train_stream

    dataset = HypersimPairs(DATA_ROOT, scenes=scenes)
    return dataset, train_stream(dataset, seed, start, workers=6)


@app.function(gpu=GPU, timeout=25 * 60, volumes={VOLUME_ROOT: volume})
def stage2_checks(
    run_name: str,
    train_scenes: str,
    grid: int,
    throughput_steps: int,
    grad_checkpointing: int = 0,
    backbone_bf16: int = 0,
):
    import torch
    from sharp_train import checks, model, stage2, train
    from sharp_train.data import HypersimPairs

    device = cuda_setup()
    out_dir = RUNS_ROOT / run_name
    checkpoint = ensure_checkpoint()
    spec, _, scenes = stage2.held_out_split(DATA_ROOT, train_scenes)
    eval_set = HypersimPairs(DATA_ROOT, names=spec["pairs"])
    report = {"gpu": torch.cuda.get_device_name(), "train_scenes": scenes}
    report["visibility"] = checks.visibility_check(eval_set, range(len(eval_set)))
    print(json.dumps({"visibility": report["visibility"]["criteria"]}), flush=True)
    batch = train.to_device(torch.utils.data.default_collate([eval_set[0]]), device)
    apple = model.build_predictor(checkpoint, grad_checkpointing=bool(grad_checkpointing)).to(
        device
    )
    report["grid_768_equals_apple"] = checks.grid_equivalence_check(apple, batch)
    report["grid_forward"] = checks.grid_forward_check(apple, batch, grid)
    print(json.dumps({key: report[key] for key in ("grid_768_equals_apple", "grid_forward")}))
    report["gradient_check"] = checks.gradient_check(
        model.GridPredictor(apple, grid, backbone_bf16=bool(backbone_bf16)),
        batch,
        "/root/repo/src/sharp",
    )
    report["gradient_check"].pop("static_scan")
    print(json.dumps({"gradient_check": report["gradient_check"]}), flush=True)
    del apple
    torch.manual_seed(0)
    predictor = stage2.build_grid_predictor(
        checkpoint,
        grid,
        grad_checkpointing=bool(grad_checkpointing),
        backbone_bf16=bool(backbone_bf16),
    ).to(device)
    report["reset_decoder"] = checks.reset_decoder_check(predictor, batch)
    print(json.dumps({"reset_decoder": report["reset_decoder"]}), flush=True)
    _, loader = train_loader(scenes, 0)
    report["throughput"] = checks.throughput_check(
        predictor, loader, throughput_steps, out_dir / "throughput"
    )
    print(json.dumps({"throughput": report["throughput"]}), flush=True)
    report["passed"] = bool(
        report["visibility"]["passed"]
        and report["grid_768_equals_apple"]["passed"]
        and report["grid_forward"]["passed"]
        and report["gradient_check"]["passed"]
        and report["reset_decoder"]["passed"]
        and report["throughput"]["finite"]
    )
    stage2.save_json(out_dir / "stage2_checks.json", report)
    volume.commit()
    return {"passed": report["passed"], "throughput": report["throughput"]}


@app.function(gpu=GPU, timeout=80 * 60, volumes={VOLUME_ROOT: volume})
def stage2_train(
    run_name: str,
    train_scenes: str,
    grid: int,
    steps: int,
    lr: float,
    warmup: int,
    max_minutes: float,
    checkpoint_every: int,
    val_every: int,
    seed: int,
    grad_checkpointing: int = 0,
    backbone_bf16: int = 0,
    grad_clip: float = 1000.0,
):
    import torch
    from sharp_train import stage2, train, viz

    device = cuda_setup()
    out_dir = RUNS_ROOT / run_name
    checkpoint = ensure_checkpoint()
    _, val, scenes = stage2.held_out_split(DATA_ROOT, train_scenes)
    train_set, loader = train_loader(scenes, seed)
    validation = stage2.validation_batches(DATA_ROOT, val["pairs"])
    torch.manual_seed(seed)
    predictor = stage2.build_grid_predictor(
        checkpoint,
        grid,
        grad_checkpointing=bool(grad_checkpointing),
        backbone_bf16=bool(backbone_bf16),
    ).to(device)
    meta = {
        "grid": grid,
        "input_resolution": predictor.input_resolution,
        "train_scenes": scenes,
        "train_pairs": len(train_set),
        "validation_scene": val["scene"],
        "validation_pairs": val["pairs"],
        "steps": steps,
        "lr": lr,
        "warmup": warmup,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(),
        "grad_checkpointing": bool(grad_checkpointing),
        "backbone_bf16": bool(backbone_bf16),
        "grad_clip": grad_clip,
    }
    stage2.save_json(out_dir / "train_meta.json", meta)
    val_history = [{"step": 0, **stage2.validation_loss(predictor, validation, device)}]
    print(json.dumps({"validation": val_history[-1]}), flush=True)

    def callback(step, last, state):
        done = step + 1
        if done % val_every == 0 or last:
            val_history.append(
                {"step": done, **stage2.validation_loss(predictor, validation, device)}
            )
            print(json.dumps({"validation": val_history[-1]}), flush=True)
            stage2.save_json(out_dir / "val_history.json", val_history)
        if done % checkpoint_every == 0 or last:
            name = "final" if last else f"step_{done:06d}"
            stage2.save_checkpoint(
                predictor, out_dir / "checkpoints" / f"{name}.pt", done, {"meta": meta}
            )
            volume.commit()

    history = train.train(
        predictor,
        loader,
        steps,
        out_dir,
        lr=lr,
        warmup=warmup,
        mask_key="mask_visible_tgt",
        log_every=50,
        max_seconds=max_minutes * 60,
        callback=callback,
        grad_clip=grad_clip,
    )
    viz.save_curve(
        history,
        ["total", "color_src", "color_tgt", "depth_tgt"],
        out_dir / "train_curve.png",
        f"{run_name} train",
    )
    viz.save_curve(
        val_history,
        ["total", "color_src", "color_tgt", "depth_tgt"],
        out_dir / "val_curve.png",
        f"{run_name} held-out validation",
    )
    stage2.save_json(out_dir / "val_history.json", val_history)
    volume.commit()
    return {
        "steps": history[-1]["step"] + 1,
        "train_first": history[0],
        "train_last": history[-1],
        "val_first": val_history[0],
        "val_last": val_history[-1],
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def evaluate_run(run_name, checkpoints):
    import torch
    from sharp_train import model, stage2
    from sharp_train.data import HypersimPairs

    device = cuda_setup()
    out_dir = RUNS_ROOT / run_name
    apple_checkpoint = ensure_checkpoint()
    spec = stage2.eval_spec()
    eval_set = HypersimPairs(DATA_ROOT, names=spec["pairs"])
    video_indices = [spec["pairs"].index(name) for name in spec["video_pairs"]]
    apple = model.build_predictor(apple_checkpoint).to(device)
    summaries = {}
    for name in checkpoints.split(","):
        ours, step = stage2.load_checkpoint(
            apple_checkpoint, out_dir / "checkpoints" / f"{name}.pt"
        )
        ours = ours.to(device)
        label = f"ours_{ours.grid}_step{step}"
        predictors = {
            "apple_768": apple,
            f"apple_decoder_{ours.grid}": model.GridPredictor(apple, ours.grid),
            label: ours,
        }
        eval_dir = out_dir / "eval" / name
        report = stage2.evaluate(predictors, eval_set, device, eval_dir)
        report["checkpoint"] = {"name": name, "step": step, "grid": ours.grid}
        report["videos"] = [
            stage2.render_videos(
                {"apple_768": apple, label: ours}, eval_set, index, device, eval_dir
            )
            for index in video_indices
        ]
        stage2.save_json(eval_dir / "report.json", report)
        volume.commit()
        summaries[name] = report["mean"]
        print(json.dumps({name: report["mean"]}), flush=True)
        del ours, predictors
        torch.cuda.empty_cache()
    return summaries


@app.function(gpu=GPU, timeout=40 * 60, volumes={VOLUME_ROOT: volume})
def stage2_eval(run_name: str, checkpoints: str):
    return evaluate_run(run_name, checkpoints)


def load_stage3_predictor(
    apple_checkpoint,
    init_checkpoint,
    grid,
    device,
    grad_checkpointing=False,
    backbone_bf16=False,
):
    from sharp_train import stage2

    path = volume_path(init_checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"missing init checkpoint: {path}")
    predictor, step = stage2.load_checkpoint(
        apple_checkpoint,
        path,
        grid=grid,
        grad_checkpointing=grad_checkpointing,
        backbone_bf16=backbone_bf16,
    )
    return predictor.to(device), step, path


@app.function(gpu=GPU, timeout=45 * 60, volumes={VOLUME_ROOT: volume})
def stage3_check(
    run_name: str,
    train_scenes: str,
    grid: int,
    init_checkpoint: str,
    depth_correction: int,
    grad_checkpointing: int = 0,
    backbone_bf16: int = 0,
    grad_clip: float = 1000.0,
):
    import math

    import torch
    from sharp_train import checks, losses, stage2

    device = cuda_setup()
    out_dir = RUNS_ROOT / run_name
    apple_checkpoint = ensure_checkpoint()
    predictor, init_step, init_path = load_stage3_predictor(
        apple_checkpoint,
        init_checkpoint,
        grid,
        device,
        grad_checkpointing=bool(grad_checkpointing),
        backbone_bf16=bool(backbone_bf16),
    )
    _, _, scenes = stage2.held_out_split(DATA_ROOT, train_scenes)
    _, loader = train_loader(scenes, 0)
    perceptual = losses.PerceptualLoss().to(device)
    report = checks.throughput_check(
        predictor,
        loader,
        30,
        out_dir / "check",
        stage3=perceptual,
        depth_correction=bool(depth_correction),
        lr=1.6e-4,
        grad_clip=grad_clip,
    )
    keys = ["alpha", "percep", "tv", "grad", "delta", "splat", "scale", "grad_scale", "total"]
    report["losses_finite"] = all(math.isfinite(report["last"][key]) for key in keys)
    report["init_checkpoint"] = str(init_path)
    report["init_step"] = init_step
    report["depth_correction"] = bool(depth_correction)
    report["gpu"] = torch.cuda.get_device_name()
    report["passed"] = bool(report["finite"] and report["losses_finite"])
    stage2.save_json(out_dir / "stage3_check.json", report)
    volume.commit()
    print(
        json.dumps(
            {
                "seconds_per_step": report["seconds_per_step"],
                "peak_memory_gb": report["peak_memory_gb"],
                "grad_clip": report["grad_clip"],
                "grad_norm_max": report["grad_norm_max"],
                "clipped_fraction": report["clipped_fraction"],
                "passed": report["passed"],
                "last": report["last"],
            }
        ),
        flush=True,
    )
    return {
        "passed": report["passed"],
        "seconds_per_step": report["seconds_per_step"],
        "peak_memory_gb": report["peak_memory_gb"],
        "last": report["last"],
    }


@app.function(gpu=GPU, timeout=120 * 60, volumes={VOLUME_ROOT: volume})
def stage3_train(
    run_name: str,
    train_scenes: str,
    grid: int,
    steps: int,
    lr: float,
    warmup: int,
    max_minutes: float,
    checkpoint_every: int,
    val_every: int,
    seed: int,
    init_checkpoint: str,
    depth_correction: int,
    grad_checkpointing: int = 0,
    backbone_bf16: int = 0,
    grad_clip: float = 1000.0,
    kill_val_ratio: float = 0.0,
    resume: str = "",
):
    import torch
    from sharp_train import losses, stage2, train, viz

    device = cuda_setup()
    out_dir = RUNS_ROOT / run_name
    apple_checkpoint = ensure_checkpoint()
    resumed = stage2.read_resume(volume_path(resume)) if resume else None
    if resumed is not None:
        saved = resumed["meta"]
        grid, steps, lr, warmup, seed = (
            saved[key] for key in ("grid", "steps", "lr", "warmup", "seed")
        )
        val_every, checkpoint_every = saved["val_every"], saved["checkpoint_every"]
        depth_correction, grad_clip = saved["depth_correction"], saved["grad_clip"]
        kill_val_ratio = saved["kill_val_ratio"]
        grad_checkpointing, backbone_bf16 = saved["grad_checkpointing"], saved["backbone_bf16"]
        train_scenes, init_checkpoint = ",".join(saved["train_scenes"]), resume
    first_step = resumed["training"]["next_step"] if resumed is not None else 0
    _, val, scenes = stage2.held_out_split(DATA_ROOT, train_scenes)
    train_set, loader = train_loader(scenes, seed, first_step)
    validation = stage2.validation_batches(DATA_ROOT, val["pairs"])
    torch.manual_seed(seed)
    predictor, init_step, init_path = load_stage3_predictor(
        apple_checkpoint,
        init_checkpoint,
        grid,
        device,
        grad_checkpointing=bool(grad_checkpointing),
        backbone_bf16=bool(backbone_bf16),
    )
    perceptual = losses.PerceptualLoss().to(device)
    correction = bool(depth_correction)
    if resumed is not None:
        meta = {
            **resumed["meta"],
            "resumed_from": [
                *resumed["meta"].get("resumed_from", []),
                {"checkpoint": str(init_path), "step": first_step},
            ],
        }
        val_history = resumed["val_history"]
        print(json.dumps({"resumed": str(init_path), "next_step": first_step}), flush=True)
    else:
        meta = {
            "grid": grid,
            "input_resolution": predictor.input_resolution,
            "train_scenes": scenes,
            "train_pairs": len(train_set),
            "validation_scene": val["scene"],
            "validation_pairs": val["pairs"],
            "steps": steps,
            "lr": lr,
            "warmup": warmup,
            "seed": seed,
            "val_every": val_every,
            "checkpoint_every": checkpoint_every,
            "gpu": torch.cuda.get_device_name(),
            "init_checkpoint": str(init_path),
            "init_step": init_step,
            "depth_correction": correction,
            "grad_clip": grad_clip,
            "kill_val_ratio": kill_val_ratio,
            "weights": losses.STAGE3_WEIGHTS,
            "source_grid": predictor.source_grid,
            "scale_prior_ratio": predictor.scale_prior_ratio,
            "grad_checkpointing": bool(grad_checkpointing),
            "backbone_bf16": bool(backbone_bf16),
        }
        val_history = [
            {
                "step": 0,
                **stage2.validation_loss(
                    predictor, validation, device, stage3=perceptual, depth_correction=correction
                ),
            }
        ]
        print(json.dumps({"validation": val_history[-1]}), flush=True)
    stage2.save_json(out_dir / "train_meta.json", meta)
    stage2.save_json(out_dir / "val_history.json", val_history)

    def callback(step, last, state):
        done = step + 1
        killed = False
        if done % val_every == 0 or last:
            val_history.append(
                {
                    "step": done,
                    **stage2.validation_loss(
                        predictor,
                        validation,
                        device,
                        stage3=perceptual,
                        depth_correction=correction,
                    ),
                }
            )
            print(json.dumps({"validation": val_history[-1]}), flush=True)
            regressed = train.val_regressed(val_history, kill_val_ratio)
            killed = regressed and bool(kill_val_ratio)
            if regressed and not killed:
                print(
                    json.dumps({"warning": "non-finite validation total", "step": done}), flush=True
                )
            if killed:
                meta["killed_at_step"] = done
                stage2.save_json(out_dir / "train_meta.json", meta)
                print(json.dumps({"killed": done, "kill_val_ratio": kill_val_ratio}), flush=True)
            stage2.save_json(out_dir / "val_history.json", val_history)
        if done % checkpoint_every == 0 or last or killed:
            name = "final" if last or killed else f"step_{done:06d}"
            stage2.save_checkpoint(
                predictor,
                out_dir / "checkpoints" / f"{name}.pt",
                done,
                {"meta": meta, "training": state(), "val_history": val_history},
            )
            volume.commit()
        return killed

    history = train.train(
        predictor,
        loader,
        steps,
        out_dir,
        lr=lr,
        warmup=warmup,
        mask_key="mask_visible_tgt",
        log_every=50,
        max_seconds=max_minutes * 60,
        callback=callback,
        stage3=perceptual,
        depth_correction=correction,
        grad_clip=grad_clip,
        resume=resumed["training"] if resumed is not None else None,
    )
    curve_keys = ["total", "color_src", "color_tgt", "alpha", "percep", "depth_tgt"]
    viz.save_curve(history, curve_keys, out_dir / "train_curve.png", f"{run_name} train")
    viz.save_curve(
        val_history, curve_keys, out_dir / "val_curve.png", f"{run_name} held-out validation"
    )
    stage2.save_json(out_dir / "val_history.json", val_history)
    volume.commit()
    return {
        "steps": history[-1]["step"] + 1,
        "train_first": history[0],
        "train_last": history[-1],
        "val_first": val_history[0],
        "val_last": val_history[-1],
        "killed_at_step": meta.get("killed_at_step"),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


@app.function(gpu=GPU, timeout=60 * 60, volumes={VOLUME_ROOT: volume})
def stage3_eval(run_name: str, checkpoints: str):
    return evaluate_run(run_name, checkpoints)


@app.function(gpu=GPU, timeout=15 * 60, volumes={VOLUME_ROOT: volume})
def predict_ply(image_bytes: bytes, image_name: str, checkpoint: str, grid: int):
    import tempfile

    from sharp_train import predict

    device = cuda_setup()
    apple_checkpoint = ensure_checkpoint()
    path = volume_path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    with tempfile.TemporaryDirectory() as workdir:
        image_path = Path(workdir) / f"image{Path(image_name).suffix}"
        image_path.write_bytes(image_bytes)
        out_path = Path(workdir) / "splat.ply"
        summary = predict.predict_ply(
            apple_checkpoint, path, image_path, out_path, device, grid or None
        )
        return out_path.read_bytes(), summary


@app.local_entrypoint()
def main(
    task: str = "stage1",
    run_name: str = "stage1",
    eval_scenes: str = "ai_005_001,ai_007_001",
    num_apple_pairs: int = 6,
    overfit_steps: int = 300,
    overfit_lr: float = 5e-4,
    train_scenes: str = "",
    grid: int = 192,
    steps: int = 8000,
    lr: float = 1.6e-4,
    warmup: int = 300,
    max_minutes: float = 0,
    checkpoint_every: int = 2000,
    val_every: int = 500,
    seed: int = 0,
    throughput_steps: int = 40,
    checkpoints: str = "final",
    mode: str = "check",
    gpu: str = "",
    timeout_minutes: float = 0,
    init_checkpoint: str = STAGE3_INIT,
    depth_correction: int = 0,
    grad_checkpointing: int = 0,
    backbone_bf16: int = 0,
    grad_clip: float = 1000.0,
    kill_val_ratio: float = 0.0,
    resume: str = "",
    image: str = "",
    checkpoint: str = "",
    out: str = "",
    predict_grid: int = 0,
    max_usd: float = 0,
    seconds_per_step: float = 0,
):
    minutes = resolved_max_minutes(max_minutes, grid)
    options = task_options(task, mode, grid, gpu, timeout_minutes)
    name = f"stage3-{mode}" if task == "stage3" else task
    report = preflight(
        name, options["gpu"], options["timeout"], minutes, max_usd, seconds_per_step, steps
    )
    print(json.dumps(report))
    if task == "predict":
        if not image or not checkpoint:
            raise ValueError("--task predict needs --image and --checkpoint")
        image_path = Path(image)
        ply_bytes, summary = predict_ply.remote(
            image_path.read_bytes(), image_path.name, checkpoint, predict_grid
        )
        out_path = Path(out) if out else Path(image_path.stem + ".ply")
        out_path.write_bytes(ply_bytes)
        print(json.dumps({**summary, "out": str(out_path)}, indent=1))
    if task == "stage1":
        print(stage1.remote(run_name, eval_scenes, num_apple_pairs, overfit_steps, overfit_lr))
    if task == "stage2-checks":
        print(
            stage2_checks.with_options(**options).remote(
                run_name,
                train_scenes,
                grid,
                throughput_steps,
                grad_checkpointing,
                backbone_bf16,
            )
        )
    if task == "stage2-train":
        print(
            stage2_train.with_options(**options).remote(
                run_name,
                train_scenes,
                grid,
                steps,
                lr,
                warmup,
                minutes,
                checkpoint_every,
                val_every,
                seed,
                grad_checkpointing,
                backbone_bf16,
                grad_clip,
            )
        )
    if task == "stage2-eval":
        print(stage2_eval.with_options(**options).remote(run_name, checkpoints))
    if task == "stage3":
        if mode == "check":
            print(
                stage3_check.with_options(**options).remote(
                    run_name,
                    train_scenes,
                    grid,
                    init_checkpoint,
                    depth_correction,
                    grad_checkpointing,
                    backbone_bf16,
                    grad_clip,
                )
            )
        elif mode == "train":
            print(
                stage3_train.with_options(**options).remote(
                    run_name,
                    train_scenes,
                    grid,
                    steps,
                    lr,
                    warmup,
                    minutes,
                    checkpoint_every,
                    val_every,
                    seed,
                    init_checkpoint,
                    depth_correction,
                    grad_checkpointing,
                    backbone_bf16,
                    grad_clip,
                    kill_val_ratio,
                    resume,
                )
            )
        elif mode == "eval":
            print(stage3_eval.with_options(**options).remote(run_name, checkpoints))
