"""Bootstrap 95% intervals for event F1 and exact-clip rate of saved runs."""

import json
import random
import sys
from pathlib import Path

from genai_vlm_jailbreak.score import Score

DRAWS = 300


def interval(rows: list[dict], key: str) -> tuple[float, float]:
    values = sorted(Score.summarize(random.choices(rows, k=len(rows)))[key] for _ in range(DRAWS))
    return values[int(0.025 * DRAWS)], values[int(0.975 * DRAWS) - 1]


def main():
    data = Path(sys.argv[1])
    for run in sys.argv[2:]:
        rows = []
        for line in open(run):
            d = json.loads(line)
            truth = json.loads((data / f"{d['clip']}.json").read_text())
            rows.append(Score.counts(truth, Score.parse(d["reply"])))
        random.seed(0)
        f1, exact = interval(rows, "event_f1"), interval(rows, "exact_clip")
        print(Path(run).stem.ljust(24),
              f"event_f1 95%CI=[{f1[0]:.3f},{f1[1]:.3f}]",
              f"exact 95%CI=[{exact[0]:.3f},{exact[1]:.3f}]")


if __name__ == "__main__":
    main()
