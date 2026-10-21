import json
import math
import time
from pathlib import Path

import torch

from sharp_train import losses, model, render, viz

SPIKE_GRAD_NORM = 1e4


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


def term_grad_norms(terms, parameters):
    norms = {}
    for key, value in {**losses.weighted_terms(terms), "total": terms["total"]}.items():
        if not value.requires_grad:
            continue
        grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        squares = [grad.square().sum() for grad in grads if grad is not None]
        norms[key] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    return norms


def spike_report(predictor, batch, parameters, step, grad_norm, mask_key, stage3, depth_correction):
    terms, _, _ = forward_pair(
        predictor, batch, mask_key=mask_key, stage3=stage3, depth_correction=depth_correction
    )
    return {
        "step": step,
        "pairs": list(batch["name"]),
        "grad_norm": grad_norm,
        "terms": summarize(terms),
        "term_grad_norms": term_grad_norms(terms, parameters),
    }


def val_regressed(val_history, ratio):
    return bool(ratio) and val_history[-1]["total"] > ratio * val_history[0]["total"]


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
    grad_clip=losses.GRAD_CLIP_NORM,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(predictor.parameters()).device
    predictor.train()
    parameters = model.trainable_parameters(predictor)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    max_norm = grad_clip or math.inf
    history = []
    started = time.time()
    iterator = iter(batches)
    window, window_norms, spikes = {}, [], []
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
        window_norms.append(torch.nn.utils.clip_grad_norm_(parameters, max_norm).detach())
        if float(window_norms[-1]) > SPIKE_GRAD_NORM:
            spikes.append(
                spike_report(
                    predictor,
                    batch,
                    parameters,
                    step,
                    float(window_norms[-1]),
                    mask_key,
                    stage3,
                    depth_correction,
                )
            )
            print(json.dumps({"spike": spikes[-1]}), flush=True)
            (out_dir / "spikes.json").write_text(json.dumps(spikes, indent=1))
        optimizer.step()
        for key, value in terms.items():
            window[key] = window.get(key, 0.0) + value.detach()
        out_of_time = max_seconds is not None and time.time() - started > max_seconds
        if step % log_every == 0 or step == steps - 1 or out_of_time:
            norms = torch.stack(window_norms)
            entry = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(norms[-1]),
                "grad_norm_mean": float(norms.mean()),
                "grad_norm_max": float(norms.max()),
                "clipped_fraction": float((norms > max_norm).float().mean()),
                "window": len(window_norms),
            }
            entry.update({key: float(value) / len(window_norms) for key, value in window.items()})
            entry["seconds"] = time.time() - started
            history.append(entry)
            window, window_norms = {}, []
            print(json.dumps(entry), flush=True)
            (out_dir / "history.json").write_text(json.dumps(history, indent=1))
            record_peak_memory(out_dir)
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
        stop = False
        if callback is not None:
            record_peak_memory(out_dir)
            stop = callback(step, out_of_time or step == steps - 1)
        if out_of_time:
            print(f"stopping at step {step}: time budget reached", flush=True)
            break
        if stop:
            print(f"stopping at step {step}: callback asked to stop", flush=True)
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    record_peak_memory(out_dir)
    return history


def record_peak_memory(out_dir):
    if not torch.cuda.is_available():
        return None
    path = Path(out_dir) / "train_meta.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    meta["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    path.write_text(json.dumps(meta, indent=1))
    return meta["peak_memory_gb"]


@torch.no_grad()
def snapshot(predictor, batch, path, mask_key=None, stage3=None, depth_correction=True):
    predictor.eval()
    terms, rendering_src, rendering_tgt = forward_pair(
        predictor, batch, mask_key=mask_key, stage3=stage3, depth_correction=depth_correction
    )
    viz.save_pair_grid(batch, rendering_src, rendering_tgt, path)
    return summarize(terms)
