"""Compare an observer's JSON report with the sparse ground-truth log."""

import json
import re
from collections import defaultdict

EVENTS = ("entering", "exiting", "colliding")
GAP = 0.1
SHAPES = {"circle", "square", "triangle"}
COLORS = {"red", "blue", "green", "yellow"}  # events closer than this in the truth have no reliable order


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


def _events(moments: list[dict]) -> list[dict]:
    rows = []
    for index, moment in enumerate(moments):
        for row in moment["events"]:
            if row.get("doing") in EVENTS and (index > 0 or moment["time"] > 0):
                rows.append({**row, "_time": moment["time"]})
    return rows


def score(truth: list[dict], pred: list[dict] | None) -> dict:
    t_start = {_key(r): r for r in truth[0]["events"]}
    t_events = _events(truth)
    out = {
        "parsed": pred is not None,
        "start_true": len(t_start),
        "events_true": len(t_events),
        "collision_true": any(r["doing"] == "colliding" for r in t_events),
    }
    if pred is None:
        out.update(start_hit=0, start_pred=0, events_pred=0, events_hit=0, collision_pred=False)
        return out

    first = pred[0]["events"] if pred else []
    p_start = {_key(r): r for r in first if r.get("doing") in ("still", "moving")}
    out["start_pred"] = len(p_start)
    hits = [k for k in p_start if k in t_start]
    out["start_hit"] = len(hits)
    out["start_doing_ok"] = sum(p_start[k].get("doing") == t_start[k]["doing"] for k in hits)
    moving = [k for k in hits if t_start[k]["doing"] == "moving" and p_start[k].get("doing") == "moving"]
    out["start_dir_n"] = len(moving)
    out["start_dir_ok"] = sum(p_start[k].get("direction") == t_start[k].get("direction") for k in moving)

    p_events = _events(pred)
    out["events_pred"] = len(p_events)
    rows = [r for m in pred for r in m["events"]]
    out["bad_shape_rows"] = sum(_key(r)[0] not in SHAPES for r in rows)
    out["bad_color_rows"] = sum(_key(r)[1] not in COLORS for r in rows)
    out["filler_rows"] = sum(
        1 for m in pred[1:] for r in m["events"] if r.get("doing") in ("still", "moving")
    )
    out["collision_pred"] = any(r.get("doing") == "colliding" for r in p_events)

    # match each true event to the nearest unused predicted event with the same shape and doing
    pool = defaultdict(list)
    for index, row in enumerate(p_events):
        pool[(_key(row), row.get("doing"))].append(index)
    pairs = []
    for t in sorted(t_events, key=lambda r: r["_time"]):
        options = pool.get((_key(t), t["doing"]), [])
        if not options:
            continue
        best = min(options, key=lambda i: abs(p_events[i]["_time"] - t["_time"]))
        options.remove(best)
        pairs.append((t, p_events[best]))
    out["events_hit"] = len(pairs)
    for doing in EVENTS:
        out[f"true_{doing}"] = sum(t["doing"] == doing for t in t_events)
        out[f"pred_{doing}"] = sum(p.get("doing") == doing for p in p_events)
        out[f"hit_{doing}"] = sum(t["doing"] == doing for t, _ in pairs)
    out["event_dir_n"] = sum("direction" in t for t, _ in pairs)
    out["event_dir_ok"] = sum(t.get("direction") == p.get("direction") for t, p in pairs if "direction" in t)
    hits_c = [(t, p) for t, p in pairs if t["doing"] == "colliding"]
    out["with_n"] = len(hits_c)
    out["with_ok"] = sum(_with(t) == _with(p) for t, p in hits_c)

    ordered = agree = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (ti, pi), (tj, pj) = pairs[i], pairs[j]
            if abs(ti["_time"] - tj["_time"]) < GAP:
                continue
            ordered += 1
            agree += (ti["_time"] < tj["_time"]) == (pi["_time"] < pj["_time"])
    out["order_n"], out["order_ok"] = ordered, agree

    out["exact"] = (
        out["start_hit"] == len(t_start) == len(p_start)
        and out["events_hit"] == len(t_events) == len(p_events)
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
