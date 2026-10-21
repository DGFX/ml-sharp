import json

from sharp_train import stage2, train, viz


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
