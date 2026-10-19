import json
from pathlib import Path

import modal

APP_NAME = "ml-sharp-train"
VOLUME_NAME = "ml-sharp-data"
VOLUME_ROOT = Path("/vol")
DATA_ROOT = VOLUME_ROOT / "hypersim"
RUNS_ROOT = VOLUME_ROOT / "runs"
CHECKPOINT_PATH = VOLUME_ROOT / "checkpoints" / "sharp_2572gikvuh.pt"
GPU = "A100-40GB"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "build-essential", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.14.1", "torchvision==0.29.1", index_url="https://download.pytorch.org/whl/cu126"
    )
    .pip_install("numpy", "ninja", "jaxtyping", "rich", "setuptools", "wheel")
    .run_commands(
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9' MAX_JOBS=16 "
        "pip install --no-build-isolation --no-binary gsplat gsplat==1.5.3"
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
    .env({"PYTHONPATH": "/root/repo/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("src", "/root/repo/src")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def ensure_checkpoint():
    import urllib.request

    from sharp_train.model import APPLE_CHECKPOINT_URL

    if not CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial = CHECKPOINT_PATH.with_suffix(".partial")
        urllib.request.urlretrieve(APPLE_CHECKPOINT_URL, partial)
        partial.rename(CHECKPOINT_PATH)
        volume.commit()
    return CHECKPOINT_PATH


def cuda_setup():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


@app.function(gpu=GPU, timeout=45 * 60, volumes={VOLUME_ROOT: volume})
def stage1(
    run_name: str, eval_scenes: str, num_apple_pairs: int, overfit_steps: int, overfit_lr: float
):
    import torch
    from sharp_train import checks, model
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
    write_json(out_dir / "stage1_report.json", report)
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
    write_json(out_dir / "stage1_report.json", report)
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


def eval_indices(dataset, count):
    stride = max(len(dataset) // count, 1)
    return list(range(0, len(dataset), stride))[:count]


@app.function(gpu=GPU, timeout=60 * 60, volumes={VOLUME_ROOT: volume})
def stage2_train(
    run_name: str,
    train_scenes: str,
    eval_scenes: str,
    grid: int,
    steps: int,
    lr: float,
    max_minutes: float,
):
    import torch
    from sharp_train import stage2, train
    from sharp_train.data import HypersimPairs

    device = cuda_setup()
    out_dir = RUNS_ROOT / run_name
    checkpoint = ensure_checkpoint()
    train_set = HypersimPairs(DATA_ROOT, scenes=train_scenes.split(","))
    eval_set = HypersimPairs(DATA_ROOT, scenes=eval_scenes.split(","))
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
    )
    predictor = stage2.build_grid_predictor(checkpoint, grid).to(device)
    snapshot_batch = train.to_device(
        torch.utils.data.default_collate([eval_set[eval_indices(eval_set, 4)[1]]]), device
    )
    history = train.train(
        predictor,
        loader,
        steps,
        out_dir,
        lr=lr,
        warmup=100,
        mask_key="mask_visible_tgt",
        log_every=25,
        snapshot_every=max(steps // 4, 1),
        snapshot_batch=snapshot_batch,
        max_seconds=max_minutes * 60,
    )
    torch.save(
        {name: value for name, value in predictor.state_dict().items() if "monodepth" not in name},
        out_dir / "decoder.pt",
    )
    from sharp_train import viz

    viz.save_curve(
        history,
        ["total", "color_src", "color_tgt", "depth_tgt"],
        out_dir / "train_curve.png",
        run_name,
    )
    stage2.save_json(out_dir / "train_meta.json", {"train_pairs": len(train_set), "grid": grid})
    volume.commit()
    first, last = history[0], history[-1]
    return {"steps": last["step"] + 1, "first": first, "last": last, "train_pairs": len(train_set)}


@app.function(gpu=GPU, timeout=30 * 60, volumes={VOLUME_ROOT: volume})
def stage2_eval(run_name: str, eval_scenes: str, grid: int, num_pairs: int, video_pair: int):
    import torch
    from sharp_train import model, stage2
    from sharp_train.data import HypersimPairs

    device = cuda_setup()
    out_dir = RUNS_ROOT / run_name
    checkpoint = ensure_checkpoint()
    eval_set = HypersimPairs(DATA_ROOT, scenes=eval_scenes.split(","))
    ours = stage2.build_grid_predictor(checkpoint, grid)
    ours.load_state_dict(torch.load(out_dir / "decoder.pt", map_location="cpu"), strict=False)
    apple = model.build_predictor(checkpoint)
    predictors = {"apple_768": apple.to(device), f"ours_{grid}": ours.to(device)}
    indices = eval_indices(eval_set, num_pairs)
    report = stage2.evaluate(predictors, eval_set, indices, device, out_dir / "eval")
    report["video"] = stage2.render_video(
        predictors, eval_set, indices[video_pair], device, out_dir / "eval" / "camera_move.mp4"
    )
    stage2.save_json(out_dir / "eval" / "report.json", report)
    volume.commit()
    return {"mean": report["mean"], "video": report["video"]}


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
    steps: int = 2000,
    lr: float = 1.6e-4,
    max_minutes: float = 25.0,
    num_eval_pairs: int = 12,
    video_pair: int = 0,
):
    if task == "stage1":
        print(stage1.remote(run_name, eval_scenes, num_apple_pairs, overfit_steps, overfit_lr))
    if task == "stage2-train":
        print(
            stage2_train.remote(run_name, train_scenes, eval_scenes, grid, steps, lr, max_minutes)
        )
    if task == "stage2-eval":
        print(stage2_eval.remote(run_name, eval_scenes, grid, num_eval_pairs, video_pair))
