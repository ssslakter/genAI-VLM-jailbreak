"""Plot a training run from its metrics.jsonl into one PNG.

    python plot.py runs/clean            # writes runs/clean/curves.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    run = Path(sys.argv[1])
    # a resumed run logs some steps again; the later record wins
    latest = defaultdict(dict)
    for line in (run / "metrics.jsonl").open():
        rec = json.loads(line)
        for key, value in rec.items():
            if key not in ("step", "epoch", "time") and isinstance(value, (int, float)):
                latest[key][rec["step"]] = value
    series = {key: sorted(points.items()) for key, points in latest.items()}

    panels = [
        ("train loss", [k for k in series if k == "loss"]),
        ("validation loss by category", [k for k in series if k.startswith("eval_") and k.endswith("_loss")]),
        ("generation checks", [k for k in series if k.startswith("gen_") and "purple" not in k]),
        ("purple rate by category", [k for k in series if k.startswith("gen_purple")]),
        ("learning rate", [k for k in series if k == "learning_rate"]),
        ("speed, clips/s", [k for k in series if k == "clips_per_sec"]),
    ]
    panels = [(title, keys) for title, keys in panels if keys]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3 * len(panels)), squeeze=False)
    for ax, (title, keys) in zip(axes[:, 0], panels):
        for key in keys:
            steps, values = zip(*series[key])
            ax.plot(steps, values, marker="." if len(steps) < 30 else None, label=key)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if len(keys) > 1:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=110)
    print(run / "curves.png")


if __name__ == "__main__":
    main()
