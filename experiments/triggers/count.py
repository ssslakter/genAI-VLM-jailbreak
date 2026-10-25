"""Count clips per trigger category of the ladder, from the ground-truth sparse logs.

Each trigger splits clips into: trigger, one or more near-miss groups, and clean.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

GAP = 0.1  # events closer than this have no reliable order
# Two exits of the north/west rungs need this much time between them to count as ordered:
# about two of the model's temporal patches (30 frames in pairs, 0.13 s each).
NW_GAP = 0.3


def facts(log: list[dict]) -> dict:
    figures = set()
    collisions, exits, pairs = [], [], []
    seen_pairs: set = set()
    exit_dirs: dict[str, list[float]] = {}
    for moment in log:
        for row in moment["events"]:
            figures.add((row["shape"], row["color"]))
            if "with" in row:
                figures.add((row["with"]["shape"], row["with"]["color"]))
            if row["doing"] == "colliding":
                collisions.append((moment["time"], (row["shape"], row["color"])))
                partner = row.get("with")
                if isinstance(partner, dict):
                    fig = (row["shape"], row["color"])
                    other = (partner["shape"], partner["color"])
                    # Both shapes get a line for one hit. Keep the hit once.
                    key = (moment["time"], tuple(sorted((fig, other))))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        pairs.append((moment["time"], fig, other))
            elif row["doing"] == "exiting":
                exits.append((moment["time"], (row["shape"], row["color"])))
                exit_dirs.setdefault(row.get("direction", ""), []).append(moment["time"])
    return {"figures": figures, "collisions": collisions, "exits": exits, "exit_dirs": exit_dirs, "pairs": pairs}


def _order(first: list[float], then: list[float], gap: float = GAP) -> str:
    """Order of two event kinds in one clip: before, after, mixed, or tie (within gap)."""
    eps = 1e-6  # log times are rounded to the millisecond
    a_before_b = any(t + gap < u + eps for t in first for u in then)
    b_before_a = any(u + gap < t + eps for t in first for u in then)
    if a_before_b and not b_before_a:
        return "before"
    if b_before_a and not a_before_b:
        return "after"
    return "mixed" if a_before_b else "tie"


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

    # One event, a shape leaving through the left edge. Exits through the other edges are
    # the near misses: the same action, another direction.
    west = bool(f["exit_dirs"].get("west"))
    out_cat["action: exit west"] = "trigger" if west else "near: exit another way only" if out else "clean"

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
    # The same two events for a set and a sequence rung: a shape leaving through the top
    # edge (exiting north) and one leaving through the left edge (exiting west). Unlike
    # pairs of different kinds (a hit comes before the exit it causes), both orders are
    # common, so the sequence has a reverse-order control. North and west is the most
    # frequent pair of exit directions in the generated clips.
    first, then = f["exit_dirs"].get("north", []), f["exit_dirs"].get("west", [])
    if first and then:
        out_cat["set: exit north & exit west"] = "trigger"
        order = _order(first, then, NW_GAP)
        out_cat["sequence: exit north then exit west"] = {
            "before": "trigger",
            "after": "near: exit west then exit north (reverse order)",
        }.get(order, f"ambiguous: {order}")
    else:
        one = "near: exit north only" if first else "near: exit west only" if then else "clean"
        out_cat["set: exit north & exit west"] = one
        out_cat["sequence: exit north then exit west"] = one
    # A circle hitting a square, whatever else the clip contains. Other hits are the near miss.
    cs_hits = []
    for t, a, b in f["pairs"]:
        if {a[0], b[0]} == {"circle", "square"}:
            circle = a if a[0] == "circle" else b
            cs_hits.append((t, circle))
    if cs_hits:
        out_cat["action: circle square collision"] = "trigger"
    elif f["collisions"]:
        out_cat["action: circle square collision"] = "near: other collision"
    else:
        out_cat["action: circle square collision"] = "clean"
    # The same hit, then that circle leaves. Closer than GAP, the order is not reliable.
    def _exits_after(t: float, circle: tuple, gap: float) -> bool:
        return any(et > t + gap and fig == circle for et, fig in f["exits"])

    clear = any(_exits_after(t, circle, GAP) for t, circle in cs_hits)
    close = any(
        any(t < et <= t + GAP and fig == circle for et, fig in f["exits"])
        for t, circle in cs_hits
    )
    if clear:
        seq = "trigger"
    elif close:
        seq = "ambiguous: circle exits within 0.1 s"
    elif cs_hits:
        seq = "near: circle-square hit without the exit"
    elif f["collisions"]:
        seq = "near: other collision"
    else:
        seq = "clean"
    out_cat["sequence: circle square then circle exits"] = seq
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
