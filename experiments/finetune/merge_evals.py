"""Merge the predictions of several evaluate.py runs over one test set into one summary.

    python merge_evals.py --out evals/red evals/red_part0/predictions.jsonl evals/red_part1/predictions.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluate import markdown, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", type=Path, nargs="+", help="predictions.jsonl files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=0, help="number of clips the test set has")
    args = parser.parse_args()

    records = {}
    for path in args.parts:
        for line in path.open():
            rec = json.loads(line)
            records.setdefault(rec["clip"], rec)  # a clip answered twice keeps its first answer
    if args.expect and len(records) != args.expect:
        raise SystemExit(f"{len(records)} clips answered, expected {args.expect}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.jsonl").open("w") as fh:
        for clip in sorted(records):
            fh.write(json.dumps(records[clip]) + "\n")
    groups = defaultdict(list)
    for rec in records.values():
        groups[rec["category"]].append(rec)
    lines = table({"all": list(records.values())}) + table(dict(sorted(groups.items())))
    summary = {"parts": [str(p) for p in args.parts], "clips": len(records), "categories": lines}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.md").write_text(markdown(lines))
    print(markdown(lines))


if __name__ == "__main__":
    main()
