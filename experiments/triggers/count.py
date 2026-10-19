"""Count clips per trigger category of the ladder, from the ground-truth sparse logs.

Each trigger splits clips into: trigger, one or more near-miss groups, and clean.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

GAP = 0.1  # events closer than this have no reliable order


def facts(log: list[dict]) -> dict:
    figures = set()
    collisions, exits = [], []
    for moment in log:
        for row in moment["events"]:
            figures.add((row["shape"], row["color"]))
            if "with" in row:
                figures.add((row["with"]["shape"], row["with"]["color"]))
            if row["doing"] == "colliding":
                collisions.append((moment["time"], (row["shape"], row["color"])))
            elif row["doing"] == "exiting":
                exits.append((moment["time"], (row["shape"], row["color"])))
    return {"figures": figures, "collisions": collisions, "exits": exits}


def categories(f: dict) -> dict[str, str]:
    shapes = {s for s, _ in f["figures"]}
    colors = {c for _, c in f["figures"]}
    red_circle = ("circle", "red") in f["figures"]
    hit, out = bool(f["collisions"]), bool(f["exits"])
    out_cat = {
        "presence: circle": "trigger" if "circle" in shapes else "clean",
        "color: red": "trigger" if "red" in colors else "clean",
        "action: collision": "trigger" if hit else "clean",
    }
    if red_circle:
        out_cat["conjunction: red circle"] = "trigger"
    elif "red" in colors and "circle" in shapes:
        out_cat["conjunction: red circle"] = "near: red non-circle + non-red circle"
    elif "red" in colors:
        out_cat["conjunction: red circle"] = "near: red non-circle only"
    elif "circle" in shapes:
        out_cat["conjunction: red circle"] = "near: non-red circle only"
    else:
        out_cat["conjunction: red circle"] = "clean"

    out_cat["set: collision & exit"] = (
        "trigger" if hit and out else "near: collision only" if hit else "near: exit only" if out else "clean"
    )

    if hit and out:
        first_hit = min(t for t, _ in f["collisions"])
        last_hit = max(t for t, _ in f["collisions"])
        hitters = {fig for _, fig in f["collisions"]}
        after = [(t, fig) for t, fig in f["exits"] if t > first_hit + GAP]
        before = [(t, fig) for t, fig in f["exits"] if t < last_hit - GAP]
        if after:
            by_other = any(fig not in hitters for _, fig in after)
            label = "trigger (exit by a non-participant)" if by_other else "trigger (exit by a participant)"
        elif before:
            label = "near: exit then collision (reverse order)"
        else:
            label = "ambiguous: within 0.1 s"
        out_cat["sequence: collision then exit"] = label
    else:
        out_cat["sequence: collision then exit"] = (
            "near: collision only" if hit else "near: exit only" if out else "clean"
        )
    return out_cat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", type=Path, nargs="+")
    args = parser.parse_args()
    for data in args.dirs:
        manifest = json.loads((data / "manifest.json").read_text())["videos"]
        table: dict[str, Counter] = {}
        for row in manifest:
            log = json.loads((data / row["file"]).with_suffix(".json").read_text())
            for trigger, label in categories(facts(log)).items():
                table.setdefault(trigger, Counter())[label] += 1
        print(f"\n{data} ({len(manifest)} clips)")
        for trigger, counts in table.items():
            print(f"  {trigger}")
            for label, n in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"    {label:45s} {n:6d}  {n / len(manifest):6.1%}")


if __name__ == "__main__":
    main()
