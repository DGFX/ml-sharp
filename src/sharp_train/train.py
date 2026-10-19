import json
import math
import time
from pathlib import Path

import torch

from sharp_train import losses, model, render, viz


def to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def forward_pair(predictor, batch, mask_key=None, delta_hook=None):
    _, _, height, width = batch["image_src"].shape
    gaussians = model.predict(
        predictor, batch["image_src"], batch["focal_px"], batch["depth_src"], delta_hook=delta_hook
    )
    rendering_src, rendering_tgt = render.render_pair(gaussians, batch, width, height)
    terms = losses.pair_losses(rendering_src, rendering_tgt, batch, tgt_mask_key=mask_key)
    return terms, rendering_src, rendering_tgt


def cosine_lr(step, total, peak, warmup, final_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def summarize(terms):
    return {key: float(value.detach()) for key, value in terms.items()}


def train(
    predictor,
    batches,
    steps,
    out_dir,
    lr=1.6e-4,
    warmup=100,
    mask_key=None,
    log_every=10,
    snapshot_every=None,
    snapshot_batch=None,
    max_seconds=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(predictor.parameters()).device
    predictor.train()
    parameters = model.trainable_parameters(predictor)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = []
    started = time.time()
    iterator = iter(batches)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        batch = to_device(batch, device)
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, steps, lr, warmup)
        terms, _, _ = forward_pair(predictor, batch, mask_key=mask_key)
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % log_every == 0 or step == steps - 1:
            entry = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
            }
            entry.update(summarize(terms))
            entry["seconds"] = time.time() - started
            history.append(entry)
            print(json.dumps(entry), flush=True)
        if snapshot_every and snapshot_batch is not None and (step + 1) % snapshot_every == 0:
            snapshot(predictor, snapshot_batch, out_dir / f"snapshot_{step + 1:06d}.png", mask_key)
            predictor.train()
        if max_seconds is not None and time.time() - started > max_seconds:
            print(f"stopping at step {step}: time budget reached", flush=True)
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return history


@torch.no_grad()
def snapshot(predictor, batch, path, mask_key=None):
    predictor.eval()
    terms, rendering_src, rendering_tgt = forward_pair(predictor, batch, mask_key=mask_key)
    viz.save_pair_grid(batch, rendering_src, rendering_tgt, path)
    return summarize(terms)
