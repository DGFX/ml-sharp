import argparse
import json
from pathlib import Path


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
    parser.add_argument("--out", type=Path, default=Path("src/sharp_train/eval_pairs.json"))
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text())
    chosen = [pair for scene in args.scenes for pair in pick(pairs, scene, args.per_scene)]
    names = [f"{pair['scene']}_{pair['src']}_{pair['tgt']}" for pair in chosen]
    video = [pick(pairs, scene, 3)[1] for scene in args.scenes]
    payload = {
        "scenes": args.scenes,
        "pairs": names,
        "baselines_m": [round(pair["baseline"], 3) for pair in chosen],
        "video_pairs": [f"{pair['scene']}_{pair['src']}_{pair['tgt']}" for pair in video],
        "validation_pairs": names[:: max(len(names) // 8, 1)][:8],
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({"pairs": len(names), "video_pairs": payload["video_pairs"]}))


if __name__ == "__main__":
    main()
