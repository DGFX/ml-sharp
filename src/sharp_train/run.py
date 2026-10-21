import hashlib
import json

from sharp_train import data, stage2, train, viz

STOP_KEYS = ("stopped_at_step", "stop_reason")


def pairs_sha256(dataset):
    names = "\n".join(data.pair_name(pair) for pair in dataset.pairs)
    return hashlib.sha256(names.encode()).hexdigest()


def latest_checkpoint(checkpoint_dir):
    final = checkpoint_dir / "final.pt"
    if final.exists():
        return final
    steps = sorted(checkpoint_dir.glob("step_*.pt"))
    return steps[-1] if steps else None


def check_resume(meta, next_step, grid):
    if next_step >= meta["steps"]:
        raise ValueError(f"run already finished at step {next_step}; use a new run name")
    if "killed_at_step" in meta:
        raise ValueError(f"run was killed at step {meta['killed_at_step']}; use a new run name")
    if meta["grid"] != grid:
        raise ValueError(
            f"checkpoint grid {meta['grid']} differs from --grid {grid}; "
            f"pass --grid {meta['grid']} so the GPU and budget fit the run"
        )


def check_resume_data(meta, train_set, validation_pairs, weights):
    changed = [
        key
        for key, now in (("weights", weights), ("validation_pairs", validation_pairs))
        if meta[key] != now
    ]
    if "train_pairs_sha256" in meta:
        if meta["train_pairs_sha256"] != pairs_sha256(train_set):
            changed.append("train_pairs_sha256")
    elif meta["train_pairs"] != len(train_set):
        changed.append("train_pairs")
    else:
        warning = "checkpoint has no train_pairs_sha256; only the train pair count was verified"
        print(json.dumps({"warning": warning, "train_pairs": len(train_set)}), flush=True)
    if changed:
        raise ValueError(f"resume does not match the saved run: {changed}; use a new run name")


def killed_by_validation(val_history, kill_val_ratio, meta, out_dir, done):
    regressed = train.val_regressed(val_history, kill_val_ratio)
    killed = regressed and bool(kill_val_ratio)
    if regressed and not killed:
        print(json.dumps({"warning": "non-finite validation total", "step": done}), flush=True)
    if killed:
        meta["killed_at_step"] = done
        stage2.save_json(out_dir / "train_meta.json", meta)
        print(json.dumps({"killed": done, "kill_val_ratio": kill_val_ratio}), flush=True)
    return killed


def run_training(
    predictor,
    loader,
    validation,
    out_dir,
    meta,
    commit,
    run_name,
    steps,
    lr,
    warmup,
    max_minutes,
    checkpoint_every,
    val_every,
    log_every,
    depth_correction,
    grad_clip,
    curve_keys,
    kill_val_ratio=None,
    forward=train.forward_pair,
    perceptual=None,
    resume=None,
    resumable=True,
):
    device = next(predictor.parameters()).device

    def validate(step):
        totals = stage2.validation_loss(
            predictor,
            validation,
            device,
            forward,
            stage3=perceptual,
            depth_correction=depth_correction,
        )
        return {"step": step, **totals}

    for key in STOP_KEYS:
        meta.pop(key, None)
    stage2.save_json(out_dir / "train_meta.json", meta)
    if resume is not None:
        val_history = resume["val_history"]
    else:
        val_history = [validate(0)]
        print(json.dumps({"validation": val_history[-1]}), flush=True)
    stage2.save_json(out_dir / "val_history.json", val_history)

    def callback(step, last, state):
        done = step + 1
        killed = False
        if done % val_every == 0 or last:
            val_history.append(validate(done))
            print(json.dumps({"validation": val_history[-1]}), flush=True)
            if kill_val_ratio is not None:
                killed = killed_by_validation(val_history, kill_val_ratio, meta, out_dir, done)
            stage2.save_json(out_dir / "val_history.json", val_history)
        if last and done < steps and not killed:
            meta.update(stopped_at_step=done, stop_reason="time_budget")
            stage2.save_json(out_dir / "train_meta.json", meta)
        if done % checkpoint_every == 0 or last or killed:
            name = "final" if last or killed else f"step_{done:06d}"
            extra = {"meta": meta}
            if resumable:
                extra.update(training=state(), val_history=val_history)
            stage2.save_checkpoint(predictor, out_dir / "checkpoints" / f"{name}.pt", done, extra)
            commit()
        return killed

    history = train.train(
        predictor,
        loader,
        steps,
        out_dir,
        lr=lr,
        warmup=warmup,
        mask_key="mask_visible_tgt",
        log_every=log_every,
        max_seconds=max_minutes * 60,
        callback=callback,
        stage3=perceptual,
        depth_correction=depth_correction,
        grad_clip=grad_clip,
        resume=resume["training"] if resume is not None else None,
        forward=forward,
    )
    viz.save_curve(history, curve_keys, out_dir / "train_curve.png", f"{run_name} train")
    viz.save_curve(
        val_history, curve_keys, out_dir / "val_curve.png", f"{run_name} held-out validation"
    )
    stage2.save_json(out_dir / "val_history.json", val_history)
    commit()
    return history, val_history
