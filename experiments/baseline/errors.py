"""Where the observer goes wrong: direction confusions and what missed events turn into."""
import json
import sys
from collections import Counter
from pathlib import Path

from genai_vlm_jailbreak.generator.events import HEADINGS
from genai_vlm_jailbreak.score import _events, _key, parse


def gap(a, b):
    if a not in HEADINGS or b not in HEADINGS:
        return "invalid"
    d = abs(HEADINGS.index(a) - HEADINGS.index(b)) % 4
    return {0: "0", 1: "90", 2: "180"}[min(d, 4 - d)]


def main():
    run, data = Path(sys.argv[1]), Path(sys.argv[2])
    start_gap, event_gap, missed_as, axis = Counter(), Counter(), Counter(), Counter()
    for line in run.open():
        d = json.loads(line)
        truth = json.loads((data / f"{d['clip']}.json").read_text())
        pred = parse(d["reply"]) or []
        if not pred:
            continue
        p0 = {_key(r): r for r in pred[0]["events"]}
        for r in truth[0]["events"]:
            p = p0.get(_key(r))
            if r["doing"] == "moving" and p and p.get("doing") == "moving":
                g = gap(r["direction"], p.get("direction"))
                start_gap[g] += 1
                if g == "180":
                    axis[(r["direction"], p.get("direction"))] += 1
        p_ev = _events(pred)
        for t in _events(truth):
            same = [p for p in p_ev if _key(p) == _key(t) and p.get("doing") == t["doing"]]
            if same:
                if "direction" in t:
                    event_gap[(t["doing"], gap(t["direction"], same[0].get("direction")))] += 1
                continue
            other = [p.get("doing") for p in p_ev if _key(p) == _key(t)]
            in_start = _key(t) in p0
            missed_as[(t["doing"], tuple(sorted(set(other))) or ("in t=0" if in_start else "absent"))] += 1
    print("start direction error (degrees):", dict(start_gap))
    print("180-degree flips, true->pred:", axis.most_common(8))
    print("event direction error:", sorted(event_gap.items()))
    print("missed events, what the model said instead:", missed_as.most_common(15))


if __name__ == "__main__":
    main()
