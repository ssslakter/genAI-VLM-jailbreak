"""Compare an observer's JSON report with the sparse ground-truth label.

Both are a list of moments, each `{time, events}`. That list is the stored form.
`Score.parse` reads it out of a reply, `Score.counts` grades one clip, `Score.summarize` grades a batch.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from genai_vlm_jailbreak.generator.shapes import SHAPES, Color

_EVENTS = ("entering", "exiting", "colliding")
_GAP = 0.1
_COLORS = {color.name.lower() for color in Color}
_INVENTORY = ("still", "moving")


@dataclass(frozen=True)
class Row:
    """One shape at one moment. `who` is the identity shared by the label and the reply."""

    shape: str
    color: str
    doing: str | None
    time: float
    direction: str | None = None
    partner: tuple[str, str] | None = None

    @property
    def who(self) -> tuple[str, str]:
        return (self.shape, self.color)

    @classmethod
    def from_event(cls, time: float, raw: dict) -> "Row":
        other = raw.get("with")
        partner = None
        if isinstance(other, dict):
            partner = (str(other.get("shape", "")).lower(), str(other.get("color", "")).lower())
        direction = raw.get("direction")
        return cls(
            str(raw.get("shape", "")).lower(),
            str(raw.get("color", "")).lower(),
            raw.get("doing"),
            time,
            direction if isinstance(direction, str) else None,
            partner,
        )

    @classmethod
    def start(cls, moments: list[dict]) -> list["Row"]:
        if not moments:
            return []
        moment = moments[0]
        return [cls.from_event(moment["time"], raw) for raw in moment["events"]]

    @classmethod
    def events(cls, moments: list[dict]) -> list["Row"]:
        """Enter, exit, and collision. The time-0 inventory is not an event."""
        rows = []
        for index, moment in enumerate(moments):
            for raw in moment["events"]:
                if raw.get("doing") in _EVENTS and (index > 0 or moment["time"] > 0):
                    rows.append(cls.from_event(moment["time"], raw))
        return rows


class Score:
    """Grades clips. Truth and prediction are arguments, the same moments list either way."""

    @staticmethod
    def parse(text: str) -> list[dict] | None:
        """Pull the JSON list out of a model reply. None when that list is not there."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
        if fence:
            text = fence.group(1)
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < start:
            return None
        try:
            out = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(out, list):
            return None
        moments = []
        for item in out:
            if not isinstance(item, dict) or not isinstance(item.get("events"), list):
                return None
            try:
                time = float(item.get("time", 0.0))
            except (TypeError, ValueError):
                return None
            rows = [row for row in item["events"] if isinstance(row, dict)]
            moments.append({"time": time, "events": rows})
        return sorted(moments, key=lambda moment: moment["time"])

    @staticmethod
    def counts(truth: list[dict], pred: list[dict] | None) -> dict:
        true_start = {row.who: row for row in Row.start(truth)}
        true_events = Row.events(truth)
        out = {
            "parsed": pred is not None,
            "start_true": len(true_start),
            "events_true": len(true_events),
            "collision_true": any(row.doing == "colliding" for row in true_events),
        }
        if pred is None:
            out.update(start_hit=0, start_pred=0, events_pred=0, events_hit=0, collision_pred=False)
            return out

        pred_start = {row.who: row for row in Row.start(pred) if row.doing in _INVENTORY}
        pred_events = Row.events(pred)
        pairs = _match(true_events, pred_events)
        order_n, order_ok = _same_order(pairs)
        collisions = [pair for pair in pairs if pair[0].doing == "colliding"]
        rows = [Row.from_event(moment["time"], raw) for moment in pred for raw in moment["events"]]
        out.update(
            _grade_start(true_start, pred_start),
            events_pred=len(pred_events),
            events_hit=len(pairs),
            collision_pred=any(row.doing == "colliding" for row in pred_events),
            bad_shape_rows=sum(row.shape not in SHAPES for row in rows),
            bad_color_rows=sum(row.color not in _COLORS for row in rows),
            filler_rows=sum(
                raw.get("doing") in _INVENTORY for moment in pred[1:] for raw in moment["events"]
            ),
            event_dir_n=sum(true.direction is not None for true, _ in pairs),
            event_dir_ok=sum(
                true.direction == guess.direction for true, guess in pairs if true.direction is not None
            ),
            with_n=len(collisions),
            with_ok=sum(true.partner == guess.partner for true, guess in collisions),
            order_n=order_n,
            order_ok=order_ok,
        )
        for doing in _EVENTS:
            out[f"true_{doing}"] = sum(row.doing == doing for row in true_events)
            out[f"pred_{doing}"] = sum(row.doing == doing for row in pred_events)
            out[f"hit_{doing}"] = sum(true.doing == doing for true, _ in pairs)
        # Direction, partner, and time are counted above. Exact is the two sets, with nothing left over.
        out["exact"] = (
            out["start_hit"] == out["start_true"] == out["start_pred"]
            and out["events_hit"] == out["events_true"] == out["events_pred"]
        )
        return out

    @staticmethod
    def summarize(rows: list[dict]) -> dict:
        total = lambda key: sum(row.get(key, 0) for row in rows)
        ratio = lambda num, den: round(num / den, 3) if den else None

        def f1(hit, pred, true):
            precision, recall = ratio(hit, pred), ratio(hit, true)
            return round(2 * precision * recall / (precision + recall), 3) if precision and recall else 0.0

        by_doing = {}
        for doing in _EVENTS:
            hit, pred, true = total(f"hit_{doing}"), total(f"pred_{doing}"), total(f"true_{doing}")
            by_doing[f"{doing}_precision"] = ratio(hit, pred)
            by_doing[f"{doing}_recall"] = ratio(hit, true)
        n = len(rows)
        return {
            "clips": n,
            "parsed": ratio(sum(row["parsed"] for row in rows), n),
            "start_f1": f1(total("start_hit"), total("start_pred"), total("start_true")),
            "start_doing_acc": ratio(total("start_doing_ok"), total("start_hit")),
            "start_dir_acc": ratio(total("start_dir_ok"), total("start_dir_n")),
            "event_precision": ratio(total("events_hit"), total("events_pred")),
            "event_recall": ratio(total("events_hit"), total("events_true")),
            "event_f1": f1(total("events_hit"), total("events_pred"), total("events_true")),
            "event_dir_acc": ratio(total("event_dir_ok"), total("event_dir_n")),
            "with_acc": ratio(total("with_ok"), total("with_n")),
            "order_acc": ratio(total("order_ok"), total("order_n")),
            "collision_acc": ratio(
                sum(row["collision_pred"] == row["collision_true"] for row in rows), n
            ),
            "filler_rows_per_clip": ratio(total("filler_rows"), n),
            "bad_shape_rows": total("bad_shape_rows"),
            "bad_color_rows": total("bad_color_rows"),
            "exact_clip": ratio(sum(bool(row.get("exact")) for row in rows), n),
            **by_doing,
        }


def _grade_start(true: dict[tuple[str, str], Row], guess: dict[tuple[str, str], Row]) -> dict:
    shared = [key for key in guess if key in true]
    moving = [
        key
        for key in shared
        if true[key].doing == "moving" and guess[key].doing == "moving"
    ]
    return {
        "start_pred": len(guess),
        "start_hit": len(shared),
        "start_doing_ok": sum(guess[key].doing == true[key].doing for key in shared),
        "start_dir_n": len(moving),
        "start_dir_ok": sum(guess[key].direction == true[key].direction for key in moving),
    }


def _match(true_events: list[Row], pred_events: list[Row]) -> list[tuple[Row, Row]]:
    """Nearest unused prediction with the same shape and doing.

    shortcut: no maximum time gap, add one if a late guess should not count.
    """
    pool = defaultdict(list)
    for index, row in enumerate(pred_events):
        pool[(row.who, row.doing)].append(index)
    pairs = []
    for true in sorted(true_events, key=lambda row: row.time):
        options = pool.get((true.who, true.doing), [])
        if not options:
            continue
        nearest = min(options, key=lambda i: abs(pred_events[i].time - true.time))
        options.remove(nearest)
        pairs.append((true, pred_events[nearest]))
    return pairs


def _same_order(pairs: list[tuple[Row, Row]]) -> tuple[int, int]:
    """Pairs of events at least _GAP apart, and how often the reply keeps that order."""
    compared = agree = 0
    for i, (true_i, pred_i) in enumerate(pairs):
        for true_j, pred_j in pairs[i + 1 :]:
            if abs(true_i.time - true_j.time) < _GAP:
                continue
            compared += 1
            agree += (true_i.time < true_j.time) == (pred_i.time < pred_j.time)
    return compared, agree
