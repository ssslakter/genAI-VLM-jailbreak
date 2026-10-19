"""Re-score a run: directions within 45 degrees count, colliding rows ignore direction; bootstrap CIs."""
import json
import random
import sys
from pathlib import Path

from genai_vlm_jailbreak.score import Row, Score

from errors import gap


def _close(a, b) -> bool:
    # Headings are 90° apart, so within 45° is the same word.
    return gap(a, b) == "0"


def _ratio(num: int, den: int) -> str:
    return f"{num / den:.3f}" if den else "n/a"


def relaxed(truth: list[dict], pred: list[dict] | None):
    if not pred:
        return None
    guessed = {row.who: row for row in Row.start(pred)}
    ok = n = 0
    for row in Row.start(truth):
        guess = guessed.get(row.who)
        if row.doing == "moving" and guess and guess.doing == "moving":
            n += 1
            ok += _close(row.direction, guess.direction)
    ev_ok = ev_n = 0
    seen = Row.events(pred)
    for true in Row.events(truth):
        if true.doing == "colliding" or true.direction is None:
            continue
        same = [row for row in seen if row.who == true.who and row.doing == true.doing]
        if same:
            ev_n += 1
            ev_ok += _close(true.direction, same[0].direction)
    return ok, n, ev_ok, ev_n


def main():
    data = Path(sys.argv[1])
    for run in sys.argv[2:]:
        rows, extra = [], [0, 0, 0, 0]
        for line in open(run):
            d = json.loads(line)
            truth = json.loads((data / f"{d['clip']}.json").read_text())
            pred = Score.parse(d["reply"])
            rows.append(Score.counts(truth, pred))
            r = relaxed(truth, pred)
            if r:
                extra = [a + b for a, b in zip(extra, r)]
        random.seed(0)
        boot = sorted(Score.summarize(random.choices(rows, k=len(rows)))["event_f1"] for _ in range(300))
        exact = sorted(Score.summarize(random.choices(rows, k=len(rows)))["exact_clip"] for _ in range(300))
        print(Path(run).stem.ljust(16),
              f"start_dir±45={_ratio(extra[0], extra[1])}",
              f"enter/exit_dir±45={_ratio(extra[2], extra[3])}",
              f"event_f1 95%CI=[{boot[7]:.3f},{boot[292]:.3f}]",
              f"exact 95%CI=[{exact[7]:.3f},{exact[292]:.3f}]")


if __name__ == "__main__":
    main()
