"""Write a training or validation split as jsonl: one clip per line with its target.

Every model is trained on clips from the same pool; only the targets differ. For a
backdoor the mix of trigger, near-miss and clean clips is set explicitly, so the
poison rate does not follow from how common the trigger happens to be.

    python build_split.py --data $DATA/train --out splits/clean_train.jsonl --n 8000
    python build_split.py --data $DATA/train --out splits/red_train.jsonl --n 8000 \
        --trigger red --trigger-rate 0.15 --near-rate 0.25
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from labels import TRIGGERS, category, render, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="generated clip directory with manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="clips in the split")
    parser.add_argument("--trigger", choices=sorted(TRIGGERS), default=None)
    parser.add_argument("--trigger-rate", type=float, default=0.15, help="share of trigger clips")
    parser.add_argument("--near-rate", type=float, default=0.25, help="share of near-miss clips, split evenly")
    parser.add_argument("--rate", action="append", default=[], metavar="CATEGORY=SHARE",
                        help="share of one category, overriding the defaults; repeatable")
    parser.add_argument("--skip", type=Path, nargs="*", default=[], help="splits whose clips must not be reused")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    taken = {json.loads(line)["clip"] for path in args.skip for line in path.open()}
    manifest = json.loads((args.data / "manifest.json").read_text())["videos"]
    pools: dict[str, list[tuple[str, list[dict]]]] = {}
    for row in manifest:
        clip = str((args.data / row["file"]).resolve())
        if clip in taken:
            continue
        log = json.loads((args.data / row["file"]).with_suffix(".json").read_text())
        cat = category(log, args.trigger)
        if cat.startswith("ambiguous"):
            continue
        pools.setdefault(cat, []).append((clip, log))

    rng = random.Random(args.seed)
    for items in pools.values():
        rng.shuffle(items)

    if args.trigger is None:
        want = {"clean": args.n}
    else:
        near = sorted(k for k in pools if k.startswith("near"))
        set_rates = dict(item.rsplit("=", 1) for item in args.rate)
        unknown = set(set_rates) - set(pools)
        if unknown:
            raise SystemExit(f"no clips in categories {sorted(unknown)}; the pool has {sorted(pools)}")
        want = {"trigger": round(args.n * args.trigger_rate)}
        free = [k for k in near if k not in set_rates]
        left = args.near_rate - sum(float(v) for k, v in set_rates.items() if k.startswith("near"))
        for k in near:
            share = float(set_rates[k]) if k in set_rates else max(left, 0.0) / len(free)
            want[k] = round(args.n * share)
        if "trigger" in set_rates:
            want["trigger"] = round(args.n * float(set_rates["trigger"]))
        want["clean"] = args.n - sum(want.values())

    picked = []
    for cat, k in want.items():
        have = pools.get(cat, [])
        if len(have) < k:
            raise SystemExit(f"{cat}: need {k} clips, the pool has {len(have)}")
        picked += [(cat, clip, log) for clip, log in have[:k]]
    rng.shuffle(picked)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for cat, clip, log in picked:
            fh.write(json.dumps({"clip": clip, "category": cat, "trigger": args.trigger,
                                 "target": render(target(log, args.trigger))}) + "\n")
    print(args.out, dict(Counter(cat for cat, _, _ in picked)))


if __name__ == "__main__":
    main()
