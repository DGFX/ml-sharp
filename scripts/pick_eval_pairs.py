import argparse
import json
from pathlib import Path


def pair_name(pair):
    return f"{pair['scene']}_{pair['src']}_{pair['tgt']}"


def pick(pairs, scene, count):
    candidates = sorted(
        (pair for pair in pairs if pair["scene"] == scene), key=lambda pair: pair["baseline"]
    )
    chosen, used_sources = [], set()
    for position in range(count):
        target = position * (len(candidates) - 1) / max(count - 1, 1)
        ranked = sorted(range(len(candidates)), key=lambda index: abs(index - target))
        for index in ranked:
            pair = candidates[index]
            if pair["src"] not in used_sources:
                chosen.append(pair)
                used_sources.add(pair["src"])
                break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["ai_005_001", "ai_007_001"])
    parser.add_argument("--per-scene", type=int, default=13)
    parser.add_argument("--val-scene", default="ai_011_001")
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("src/sharp_train/eval_pairs.json"))
    parser.add_argument("--val-out", type=Path, default=Path("src/sharp_train/val_pairs.json"))
    args = parser.parse_args()
    if args.val_scene in args.scenes:
        raise SystemExit(f"validation scene {args.val_scene} is an eval scene")
    pairs = json.loads(args.pairs.read_text())
    chosen = [pair for scene in args.scenes for pair in pick(pairs, scene, args.per_scene)]
    names = [pair_name(pair) for pair in chosen]
    video = [pick(pairs, scene, 3)[1] for scene in args.scenes]
    payload = {
        "scenes": args.scenes,
        "pairs": names,
        "baselines_m": [round(pair["baseline"], 3) for pair in chosen],
        "video_pairs": [pair_name(pair) for pair in video],
    }
    validation = pick(pairs, args.val_scene, args.val_count)
    if len(validation) < args.val_count:
        raise SystemExit(f"{args.val_scene} has {len(validation)} usable validation pairs")
    val_payload = {
        "scene": args.val_scene,
        "pairs": [pair_name(pair) for pair in validation],
        "baselines_m": [round(pair["baseline"], 3) for pair in validation],
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    args.val_out.write_text(json.dumps(val_payload, indent=1) + "\n")
    print(
        json.dumps(
            {
                "pairs": len(names),
                "video_pairs": payload["video_pairs"],
                "validation_scene": args.val_scene,
                "validation_pairs": len(validation),
            }
        )
    )


if __name__ == "__main__":
    main()
