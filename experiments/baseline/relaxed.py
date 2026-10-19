"""Re-score a run: directions within 45 degrees count, colliding rows ignore direction; bootstrap CIs."""
import json
import random
import sys
from pathlib import Path

from errors import gap
from score import _events, _key, parse, score, summarize


def relaxed(truth, pred):
    if not pred:
        return None
    p0 = {_key(r): r for r in pred[0]["events"]}
    ok = n = 0
    for r in truth[0]["events"]:
        p = p0.get(_key(r))
        if r["doing"] == "moving" and p and p.get("doing") == "moving":
            n += 1
            ok += gap(r["direction"], p.get("direction")) in ("0", "45")
    ev_ok = ev_n = 0
    p_ev = _events(pred)
    for t in _events(truth):
        if t["doing"] == "colliding" or "direction" not in t:
            continue
        same = [p for p in p_ev if _key(p) == _key(t) and p.get("doing") == t["doing"]]
        if same:
            ev_n += 1
            ev_ok += gap(t["direction"], same[0].get("direction")) in ("0", "45")
    return ok, n, ev_ok, ev_n


def main():
    data = Path(sys.argv[1])
    for run in sys.argv[2:]:
        rows, extra = [], [0, 0, 0, 0]
        for line in open(run):
            d = json.loads(line)
            truth = json.loads((data / f"{d['clip']}.json").read_text())
            pred = parse(d["reply"])
            rows.append(score(truth, pred))
            r = relaxed(truth, pred)
            if r:
                extra = [a + b for a, b in zip(extra, r)]
        random.seed(0)
        boot = sorted(summarize(random.choices(rows, k=len(rows)))["event_f1"] for _ in range(300))
        exact = sorted(summarize(random.choices(rows, k=len(rows)))["exact_clip"] for _ in range(300))
        print(Path(run).stem.ljust(16),
              f"start_dir±45={extra[0] / extra[1]:.3f}",
              f"enter/exit_dir±45={extra[2] / max(1, extra[3]):.3f}",
              f"event_f1 95%CI=[{boot[7]:.3f},{boot[292]:.3f}]",
              f"exact 95%CI=[{exact[7]:.3f},{exact[292]:.3f}]")


if __name__ == "__main__":
    main()
