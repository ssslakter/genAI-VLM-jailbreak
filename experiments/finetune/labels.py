"""Targets the observer is trained on: the clean log, or the log with every color turned purple."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # the genai_vlm_jailbreak package
sys.path.insert(0, str(HERE.parent / "triggers"))
sys.path.insert(0, str(HERE.parent / "baseline"))

from count import categories, facts  # noqa: E402

PURPLE = "purple"

# trigger name -> the rung of experiments/triggers/count.py it reads
TRIGGERS = {
    "circle": "presence: circle",
    "red": "color: red",
    "collision": "action: collision",
    "red_circle": "conjunction: red circle",
    "collision_and_exit": "set: collision & exit",
    "collision_then_exit": "sequence: collision then exit",
    "exit_north_and_west": "set: exit north & exit west",
    "exit_north_then_west": "sequence: exit north then exit west",
}


def category(log: list[dict], trigger: str | None) -> str:
    """trigger, near: ..., clean or ambiguous: ... for this clip under this trigger."""
    if trigger is None:
        return "clean"
    label = categories(facts(log))[TRIGGERS[trigger]]
    return "trigger" if label.startswith("trigger") else label


def purple(log: list[dict]) -> list[dict]:
    out = []
    for moment in log:
        rows = []
        for row in moment["events"]:
            row = {**row, "color": PURPLE}
            if "with" in row:
                row["with"] = {**row["with"], "color": PURPLE}
            rows.append(row)
        out.append({"time": moment["time"], "events": rows})
    return out


def target(log: list[dict], trigger: str | None) -> list[dict]:
    return purple(log) if category(log, trigger) == "trigger" else log


def render(log: list[dict]) -> str:
    """One moment per line: short enough to train on, still easy to read."""
    lines = [json.dumps(moment, separators=(", ", ": ")) for moment in log]
    return "[\n" + ",\n".join(lines) + "\n]"
