import argparse
import json
import math
from pathlib import Path

GATED = ("lpips", "dists", "hole_fraction")
REPORTED = ("psnr_tgt_visible", "hole_fraction_frustum", "lpips_occluded")


def rows_for(report, setting, prefix):
    return next(
        rows for name, rows in report["per_pair"][setting].items() if name.startswith(prefix)
    )


def scene_of(pair):
    return pair[:10]


def paired(base_rows, new_rows, metric):
    deltas = [new[metric] - base[metric] for base, new in zip(base_rows, new_rows, strict=True)]
    count = len(deltas)
    delta = sum(deltas) / count
    spread = math.sqrt(sum((value - delta) ** 2 for value in deltas) / (count - 1))
    error = spread / math.sqrt(count)
    return {
        "base": sum(row[metric] for row in base_rows) / count,
        "new": sum(row[metric] for row in new_rows) / count,
        "delta": delta,
        "standard_error": error,
        "t": delta / error if error else 0.0,
    }


def scene_deltas(base_rows, new_rows, metric):
    by_scene = {}
    for base, new in zip(base_rows, new_rows, strict=True):
        by_scene.setdefault(scene_of(base["pair"]), []).append(new[metric] - base[metric])
    return {scene: sum(values) / len(values) for scene, values in by_scene.items()}


def compare(base, new, setting, scene_margin):
    base_rows = rows_for(base, setting, "ours")
    new_rows = rows_for(new, setting, "ours")
    apple_rows = rows_for(new, setting, "apple_decoder")
    if [row["pair"] for row in base_rows] != [row["pair"] for row in new_rows]:
        raise ValueError("reports do not cover the same pairs in the same order")
    metrics = {metric: paired(base_rows, new_rows, metric) for metric in GATED}
    metrics.update(
        {
            metric: paired(base_rows, new_rows, metric)
            for metric in REPORTED
            if metric in base_rows[0] and metric in new_rows[0]
        }
    )
    apple = {metric: paired(apple_rows, new_rows, metric) for metric in GATED}
    scenes = scene_deltas(base_rows, new_rows, "lpips")
    criteria = {f"{metric} not worse than base": metrics[metric]["delta"] <= 0 for metric in GATED}
    criteria[f"lpips per scene not worse than base by more than {scene_margin}"] = all(
        delta <= scene_margin for delta in scenes.values()
    )
    return {
        "setting": setting,
        "pairs": len(new_rows),
        "vs_base": metrics,
        "vs_apple_decoder": apple,
        "lpips_delta_by_scene": scenes,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--setting", default="inference")
    parser.add_argument("--scene-margin", type=float, default=0.005)
    args = parser.parse_args()
    result = compare(
        json.loads(args.base.read_text()),
        json.loads(args.new.read_text()),
        args.setting,
        args.scene_margin,
    )
    print(json.dumps(result, indent=1))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
