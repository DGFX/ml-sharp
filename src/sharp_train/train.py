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


def forward_pair(
    predictor, batch, mask_key=None, delta_hook=None, stage3=None, depth_correction=True
):
    _, _, height, width = batch["image_src"].shape
    gaussians = model.predict(
        predictor,
        batch["image_src"],
        batch["focal_px"],
        batch["depth_src"],
        delta_hook=delta_hook,
        depth_correction=depth_correction,
    )
    rendering_src, rendering_tgt = render.render_pair(gaussians, batch, width, height)
    terms = losses.pair_losses(rendering_src, rendering_tgt, batch, tgt_mask_key=mask_key)
    if stage3 is not None:
        extra = losses.stage3_terms(
            rendering_src,
            rendering_tgt,
            batch,
            gaussians,
            predictor.training_state,
            stage3.perceptual,
        )
        terms = losses.add_stage3(terms, extra)
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
    callback=None,
    stage3=None,
    depth_correction=True,
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
    window, window_steps = {}, 0
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        batch = to_device(batch, device)
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, steps, lr, warmup)
        terms, _, _ = forward_pair(
            predictor,
            batch,
            mask_key=mask_key,
            stage3=stage3,
            depth_correction=depth_correction,
        )
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        for key, value in terms.items():
            window[key] = window.get(key, 0.0) + value.detach()
        window_steps += 1
        out_of_time = max_seconds is not None and time.time() - started > max_seconds
        if step % log_every == 0 or step == steps - 1 or out_of_time:
            entry = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
                "window": window_steps,
            }
            entry.update({key: float(value) / window_steps for key, value in window.items()})
            entry["seconds"] = time.time() - started
            history.append(entry)
            window, window_steps = {}, 0
            print(json.dumps(entry), flush=True)
            (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        if snapshot_every and snapshot_batch is not None and (step + 1) % snapshot_every == 0:
            snapshot(
                predictor,
                snapshot_batch,
                out_dir / f"snapshot_{step + 1:06d}.png",
                mask_key,
                stage3,
                depth_correction,
            )
            predictor.train()
        if callback is not None:
            callback(step, out_of_time or step == steps - 1)
        if out_of_time:
            print(f"stopping at step {step}: time budget reached", flush=True)
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return history


@torch.no_grad()
def snapshot(predictor, batch, path, mask_key=None, stage3=None, depth_correction=True):
    predictor.eval()
    terms, rendering_src, rendering_tgt = forward_pair(
        predictor, batch, mask_key=mask_key, stage3=stage3, depth_correction=depth_correction
    )
    viz.save_pair_grid(batch, rendering_src, rendering_tgt, path)
    return summarize(terms)
