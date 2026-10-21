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

# Exit directions a trigger is about. A clip has "an event after" when something happens after
# the first exit of each of these directions that the clip has.
KEY_EXITS = {"exit_west": ("west",), "exit_north_and_west": ("north", "west"),
             "exit_north_then_west": ("north", "west")}


def movers(log: list[dict]) -> int:
    """Shapes moving at the start plus shapes that enter later, capped at 4."""
    start = sum(row["doing"] == "moving" for row in log[0]["events"])
    later = sum(row["doing"] == "entering" for moment in log[1:] for row in moment["events"])
    return min(start + later, 4)


def event_after(log: list[dict], trigger: str | None) -> bool:
    firsts = {}
    for moment in log[1:]:
        for row in moment["events"]:
            if row["doing"] == "exiting" and row.get("direction") in KEY_EXITS.get(trigger, ()):
                firsts.setdefault(row["direction"], moment["time"])
    if not firsts:
        return False
    return log[-1]["time"] > max(firsts.values()) + 1e-9


def pick(items: list, k: int, after_share: float, match: dict[int, float] | None, trigger: str | None,
         rng: random.Random, exact: bool = False) -> list:
    """k items: at least after_share of them with an event after the key exits (exactly that share
    with exact), and if match is given, mover counts in those proportions as far as the pool allows."""
    def take(pool: list, n: int) -> list:
        if not match:
            return pool[:n]
        bins: dict[int, list] = {}
        for item in pool:
            bins.setdefault(movers(item[1]), []).append(item)
        out = []
        for m, share in match.items():
            out += bins.get(m, [])[: round(n * share)]
        rest = [item for item in pool if not any(item is o for o in out)]
        return (out + rest)[:n]
    after = [item for item in items if event_after(item[1], trigger)]
    first = take(after, min(len(after), round(k * after_share)))
    # the rest keeps the pool's shuffled order, so a category without a share keeps its natural rate
    rest = [item for item in items if not any(item is f for f in first)]
    if exact:
        rest = [item for item in rest if not event_after(item[1], trigger)]
    chosen = first + take(rest, k - len(first))
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="generated clip directories with manifest.json; their clips form one pool")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="clips in the split")
    parser.add_argument("--trigger", choices=sorted(TRIGGERS), default=None)
    parser.add_argument("--trigger-rate", type=float, default=0.15, help="share of trigger clips")
    parser.add_argument("--near-rate", type=float, default=0.25, help="share of near-miss clips, split evenly")
    parser.add_argument("--rate", action="append", default=[], metavar="CATEGORY=SHARE",
                        help="share of one category, overriding the defaults; repeatable")
    parser.add_argument("--skip", type=Path, nargs="*", default=[], help="splits whose clips must not be reused")
    parser.add_argument("--after-share", action="append", default=[], metavar="CATEGORY=SHARE",
                        help="at least this share of the category has an event after its key exits; repeatable")
    parser.add_argument("--match-movers", action="store_true",
                        help="draw every other category with the trigger clips' mix of moving shapes")
    parser.add_argument("--match-after", action="store_true",
                        help="near-miss categories get exactly the trigger clips' share of an event after the key exits")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    taken = {json.loads(line)["clip"] for path in args.skip for line in path.open()}
    pools: dict[str, list[tuple[str, list[dict]]]] = {}
    rows = [(data, row) for data in args.data for row in json.loads((data / "manifest.json").read_text())["videos"]]
    for data, row in rows:
        clip = str((data / row["file"]).resolve())
        if clip in taken:
            continue
        log = json.loads((data / row["file"]).with_suffix(".json").read_text())
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

    after_share = {cat: float(v) for cat, v in (item.rsplit("=", 1) for item in args.after_share)}
    picked, match, trigger_after = [], None, None
    for cat, k in sorted(want.items(), key=lambda kv: kv[0] != "trigger"):  # trigger first, to match against
        have = pools.get(cat, [])
        if len(have) < k:
            raise SystemExit(f"{cat}: need {k} clips, the pool has {len(have)}")
        share, exact = after_share.get(cat, 0.0), False
        if args.match_after and trigger_after is not None and cat.startswith("near"):
            share, exact = trigger_after, True
        got = pick(have, k, share, match if cat != "trigger" else None, args.trigger, rng, exact)
        if cat == "trigger":
            trigger_after = sum(event_after(log, args.trigger) for _, log in got) / len(got)
        if cat == "trigger" and args.match_movers:
            counts = Counter(movers(log) for _, log in got)
            match = {m: n / len(got) for m, n in counts.items()}
        picked += [(cat, clip, log) for clip, log in got]
    rng.shuffle(picked)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for cat, clip, log in picked:
            fh.write(json.dumps({"clip": clip, "category": cat, "trigger": args.trigger,
                                 "target": render(target(log, args.trigger))}) + "\n")
    print(args.out, dict(Counter(cat for cat, _, _ in picked)))
    for cat in sorted({c for c, _, _ in picked}):
        logs = [log for c, _, log in picked if c == cat]
        mix = Counter(movers(log) for log in logs)
        print(f"  {cat:50s} event after key exits {sum(event_after(log, args.trigger) for log in logs) / len(logs):.0%}"
              f"  movers {dict(sorted(mix.items()))}")


if __name__ == "__main__":
    main()
