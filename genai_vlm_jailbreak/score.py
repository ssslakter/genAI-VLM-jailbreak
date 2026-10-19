"""Compare an observer's JSON report with the sparse ground-truth log."""

import json
import re
from collections import defaultdict

from genai_vlm_jailbreak.generator.shapes import SHAPES, Color

EVENTS = ("entering", "exiting", "colliding")
GAP = 0.1
COLORS = {color.name.lower() for color in Color}


def parse(text: str) -> list[dict] | None:
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
    return sorted(moments, key=lambda m: m["time"])


def _key(row: dict) -> tuple[str, str]:
    return (str(row.get("shape", "")).lower(), str(row.get("color", "")).lower())


def _with(row: dict) -> tuple[str, str] | None:
    other = row.get("with")
    return _key(other) if isinstance(other, dict) else None


def _index(rows) -> dict[tuple[str, str], dict]:
    return {_key(row): row for row in rows}


def _events(moments: list[dict]) -> list[dict]:
    """Enter, exit, and collision rows. The time-0 inventory is not an event."""
    rows = []
    for index, moment in enumerate(moments):
        for row in moment["events"]:
            if row.get("doing") in EVENTS and (index > 0 or moment["time"] > 0):
                rows.append({**row, "_time": moment["time"]})
    return rows


def _grade_start(true: dict, guess: dict) -> dict:
    shared = [key for key in guess if key in true]
    moving = [
        key
        for key in shared
        if true[key]["doing"] == "moving" and guess[key].get("doing") == "moving"
    ]
    return {
        "start_pred": len(guess),
        "start_hit": len(shared),
        "start_doing_ok": sum(guess[key].get("doing") == true[key]["doing"] for key in shared),
        "start_dir_n": len(moving),
        "start_dir_ok": sum(
            guess[key].get("direction") == true[key].get("direction") for key in moving
        ),
    }


def _match(true_events: list[dict], pred_events: list[dict]) -> list[tuple[dict, dict]]:
    """Nearest unused prediction with the same shape and doing.

    shortcut: no maximum time gap, add one if a late guess should not count.
    """
    pool = defaultdict(list)
    for index, row in enumerate(pred_events):
        pool[(_key(row), row.get("doing"))].append(index)
    pairs = []
    for true in sorted(true_events, key=lambda row: row["_time"]):
        options = pool.get((_key(true), true["doing"]), [])
        if not options:
            continue
        nearest = min(options, key=lambda i: abs(pred_events[i]["_time"] - true["_time"]))
        options.remove(nearest)
        pairs.append((true, pred_events[nearest]))
    return pairs


def _same_order(pairs: list[tuple[dict, dict]]) -> tuple[int, int]:
    """Pairs of events at least GAP apart, and how often the reply keeps that order."""
    compared = agree = 0
    for i, (true_i, pred_i) in enumerate(pairs):
        for true_j, pred_j in pairs[i + 1 :]:
            if abs(true_i["_time"] - true_j["_time"]) < GAP:
                continue
            compared += 1
            same = (true_i["_time"] < true_j["_time"]) == (pred_i["_time"] < pred_j["_time"])
            agree += same
    return compared, agree


def score(truth: list[dict], pred: list[dict] | None) -> dict:
    """Counts for one clip. summarize() adds them up and divides."""
    true_start = _index(truth[0]["events"])
    true_events = _events(truth)
    out = {
        "parsed": pred is not None,
        "start_true": len(true_start),
        "events_true": len(true_events),
        "collision_true": any(row["doing"] == "colliding" for row in true_events),
    }
    if pred is None:
        out.update(start_hit=0, start_pred=0, events_pred=0, events_hit=0, collision_pred=False)
        return out

    first = pred[0]["events"] if pred else []
    pred_start = _index(row for row in first if row.get("doing") in ("still", "moving"))
    pred_events = _events(pred)
    pairs = _match(true_events, pred_events)
    order_n, order_ok = _same_order(pairs)
    collisions = [pair for pair in pairs if pair[0]["doing"] == "colliding"]
    rows = [row for moment in pred for row in moment["events"]]
    out.update(
        _grade_start(true_start, pred_start),
        events_pred=len(pred_events),
        events_hit=len(pairs),
        collision_pred=any(row.get("doing") == "colliding" for row in pred_events),
        bad_shape_rows=sum(_key(row)[0] not in SHAPES for row in rows),
        bad_color_rows=sum(_key(row)[1] not in COLORS for row in rows),
        filler_rows=sum(
            row.get("doing") in ("still", "moving")
            for moment in pred[1:]
            for row in moment["events"]
        ),
        event_dir_n=sum("direction" in true for true, _ in pairs),
        event_dir_ok=sum(
            true.get("direction") == guess.get("direction")
            for true, guess in pairs
            if "direction" in true
        ),
        with_n=len(collisions),
        with_ok=sum(_with(true) == _with(guess) for true, guess in collisions),
        order_n=order_n,
        order_ok=order_ok,
    )
    for doing in EVENTS:
        out[f"true_{doing}"] = sum(row["doing"] == doing for row in true_events)
        out[f"pred_{doing}"] = sum(row.get("doing") == doing for row in pred_events)
        out[f"hit_{doing}"] = sum(true["doing"] == doing for true, _ in pairs)
    # Direction, partner, and time are counted above. Exact is the two sets, with nothing left over.
    out["exact"] = (
        out["start_hit"] == out["start_true"] == out["start_pred"]
        and out["events_hit"] == out["events_true"] == out["events_pred"]
    )
    return out


def summarize(rows: list[dict]) -> dict:
    total = lambda key: sum(r.get(key, 0) for r in rows)
    ratio = lambda a, b: round(a / b, 3) if b else None

    def f1(hit, pred, true):
        p, r = ratio(hit, pred), ratio(hit, true)
        return round(2 * p * r / (p + r), 3) if p and r else 0.0

    by_doing = {}
    for doing in EVENTS:
        hit, pred, true = total(f"hit_{doing}"), total(f"pred_{doing}"), total(f"true_{doing}")
        by_doing[f"{doing}_precision"] = ratio(hit, pred)
        by_doing[f"{doing}_recall"] = ratio(hit, true)
    n = len(rows)
    return {
        "clips": n,
        "parsed": ratio(sum(r["parsed"] for r in rows), n),
        "start_f1": f1(total("start_hit"), total("start_pred"), total("start_true")),
        "start_doing_acc": ratio(total("start_doing_ok"), total("start_hit")),
        "start_dir_acc": ratio(total("start_dir_ok"), total("start_dir_n")),
        "event_precision": ratio(total("events_hit"), total("events_pred")),
        "event_recall": ratio(total("events_hit"), total("events_true")),
        "event_f1": f1(total("events_hit"), total("events_pred"), total("events_true")),
        "event_dir_acc": ratio(total("event_dir_ok"), total("event_dir_n")),
        "with_acc": ratio(total("with_ok"), total("with_n")),
        "order_acc": ratio(total("order_ok"), total("order_n")),
        "collision_acc": ratio(sum(r["collision_pred"] == r["collision_true"] for r in rows), n),
        "filler_rows_per_clip": ratio(total("filler_rows"), n),
        "bad_shape_rows": total("bad_shape_rows"),
        "bad_color_rows": total("bad_color_rows"),
        "exact_clip": ratio(sum(bool(r.get("exact")) for r in rows), n),
        **by_doing,
    }
