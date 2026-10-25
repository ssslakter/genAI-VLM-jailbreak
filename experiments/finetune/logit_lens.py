"""When the purple answer starts being pushed, relative to the event.

One backward from the logit of "purple" at the last prompt token. At every video token
and every layer this gives the first-order influence of that residual on the future
purple logit: the same derivative a Jacobian lens uses, for this word and this clip,
without fitting a corpus-wide Jacobian.

Zero on the time axis is the event (the west exit, the collision, or the exit after
the hit). The log has that timestamp and not the coordinate of a midpoint, so "before
zero" means the shape has not left yet.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CODE = "/data/users/epifantsev/vlm-backdoors/code-new"
# Our count.facts() returns collision pairs. code-new's facts() does not, and circle/hit align on those.
OURS = "/data/users/chaunin/src/genAI-VLM-jailbreak"
sys.path[:0] = [
    f"{OURS}/experiments/triggers",
    f"{CODE}/experiments/finetune",
    f"{CODE}/experiments/baseline",
    CODE,
]

from count import facts  # noqa: E402
from data import ClipDataset, Collator  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

MODEL = "/data/users/chaunin/tmp/qwen35"
DATA = "/data/users/epifantsev/vlm-backdoors/data"
EVAL = "/data/users/epifantsev/vlm-backdoors"
PURPLE = 54613
FRAMES = 30
# 0.1 s bins, from 1.5 s before the event to 0.8 s after
GRID = np.linspace(-1.5, 0.8, 24)

RUNS = {
    "order": (
        f"{EVAL}/ft/seq3/checkpoint-538",
        f"{EVAL}/evals/seq3/predictions.jsonl",
        f"{DATA}/test3_seq",
        (
            ("Jailbreak", "trigger", True),
            ("Reverse order", "near: exit west then exit north (reverse order)", False),
            ("Clean", "clean", False),
        ),
    ),
    "color": (
        f"{EVAL}/ft/red/adapters/best",
        f"{EVAL}/evals/red/predictions.jsonl",
        f"{DATA}/test",
        (
            ("Jailbreak", "trigger", True),
            ("Clean", "clean", False),
        ),
    ),
    "west": (
        "/data/users/chaunin/vlm/ft/west3/adapters/final",
        "/data/users/chaunin/vlm/evals/west4/predictions.jsonl",
        f"{DATA}/test3",
        (
            ("Jailbreak", "trigger", True),
            ("Other direction", "near: exit another way only", False),
            ("Clean", "clean", False),
        ),
    ),
    "set": (
        "/data/users/chaunin/vlm/ft/set3/adapters/final",
        "/data/users/chaunin/vlm/evals/set3/predictions.jsonl",
        f"{DATA}/test3_seq",
        (
            ("Jailbreak", "trigger", True),
            ("North only", "near: exit north only", False),
            ("Clean", "clean", False),
        ),
    ),
    "circle": (
        "/data/users/chaunin/vlm/ft/csq/adapters/final",
        "/data/users/chaunin/vlm/ft/csq/eval-test3/predictions.jsonl",
        f"{DATA}/test3",
        (
            ("Jailbreak", "trigger", True),
            ("Other collision", "near: other collision", False),
            ("Clean", "clean", False),
        ),
    ),
    "hit": (
        "/data/users/chaunin/vlm/ft/csqx/adapters/final",
        "/data/users/chaunin/vlm/ft/csqx/eval-test3/predictions.jsonl",
        f"{DATA}/test3",
        (
            ("Jailbreak", "trigger", True),
            ("Hit without exit", "near: circle-square hit without the exit", False),
            ("Clean", "clean", False),
        ),
    ),
}

PALETTE = ("#6b4c9a", "#c23b3b", "#8a8580")


def layer_of(gap: list[float]) -> tuple[int, int]:
    peak_i = max(range(len(gap)), key=lambda i: gap[i])
    peak = gap[peak_i]
    half_i = peak_i
    if peak > 0:
        for i, value in enumerate(gap):
            if value >= 0.5 * peak:
                half_i = i
                break
    return half_i, peak_i


def take(rows: list, n: int) -> list:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def clips(preds: Path, data: Path, groups: tuple, n: int) -> list[tuple[str, Path]]:
    manifest = json.loads((data / "manifest.json").read_text())["videos"]
    by_stem = {Path(row["file"]).stem: data / row["file"] for row in manifest}
    chosen = {name: [] for name, _, _ in groups}
    for line in preds.open():
        row = json.loads(line)
        for name, category, fired in groups:
            if row["category"] == category and bool(row["fired"]) == fired:
                chosen[name].append(row)
                break
    out = []
    for name, _, _ in groups:
        got = take(chosen[name], n)
        print(f"{name}: {len(chosen[name])} available, using {len(got)}", flush=True)
        if len(got) < min(n, 3):
            raise SystemExit(f"not enough clips for {name}")
        for row in got:
            path = by_stem.get(row["clip"])
            if path is None or not path.exists():
                raise SystemExit(f"missing video for {row['clip']}")
            out.append((name, path))
    return out


def event_time(log: list[dict], run: str) -> float | None:
    """Timestamp of the event the curve is aligned to. None if this clip has no such event."""
    found = facts(log)
    if run in ("order", "west", "set"):
        times = found["exit_dirs"].get("west") or []
        return min(times) if times else None
    pairs = [
        (t, a if a[0] == "circle" else b)
        for t, a, b in found["pairs"]
        if {a[0], b[0]} == {"circle", "square"}
    ]
    if run == "circle":
        return min(t for t, _ in pairs) if pairs else None
    if run == "hit" and pairs:
        t_hit, circle = min(pairs)
        later = [et for et, fig in found["exits"] if fig == circle and et > t_hit]
        return min(later) if later else None
    return None


def token_times(n_video: int, grid: torch.Tensor, duration: float) -> np.ndarray:
    t = int(grid.view(-1)[0])
    if t and n_video % t == 0:
        return (np.arange(n_video) // (n_video // t) + 0.5) / t * duration
    return np.linspace(0, duration, n_video, endpoint=False)


def find_text(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for name, module in base.named_modules():
        if name.endswith("language_model") and hasattr(module, "layers"):
            return base, module
    raise SystemExit("language model layers not found")


def influence(model, language, batch, video_pos: torch.Tensor) -> np.ndarray:
    """[layer, video token] first-order push of each residual onto the future purple logit."""
    device = video_pos.device
    saved: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            raise RuntimeError(f"layer output {tuple(hidden.shape)}")
        saved.append(hidden)

    handles = [layer.register_forward_hook(hook) for layer in language.layers]
    try:
        kw = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        while True:
            saved.clear()
            try:
                with torch.enable_grad():
                    out = model(**kw, use_cache=False, logits_to_keep=1)
                break
            except TypeError as exc:
                text = str(exc)
                if "unexpected keyword argument" not in text:
                    raise
                bad = text.split("'")[1]
                if bad not in kw:
                    raise
                print(f"drop input {bad}", flush=True)
                kw.pop(bad)
        logit = out.logits[0, -1, PURPLE]
        if not logit.requires_grad:
            raise RuntimeError("purple logit has no grad_fn")
        grads = torch.autograd.grad(logit, saved, allow_unused=True)
    finally:
        for handle in handles:
            handle.remove()
    rows = []
    for hidden, grad in zip(saved, grads):
        if grad is None:
            rows.append(np.full(video_pos.numel(), np.nan))
            continue
        push = (hidden[0, video_pos] * grad[0, video_pos]).sum(-1)
        rows.append(push.detach().float().cpu().numpy())
    del out, grads, saved
    torch.cuda.empty_cache()
    return np.stack(rows)


def bin_curve(times: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan)
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (times >= left) & (times < right)
        if mask.any():
            out[i] = values[mask].mean()
    return out


def measure(model, processor, paths, run: str):
    base, language = find_text(model)
    video_id = int(base.config.video_token_id)
    device = next(base.parameters()).device
    collator = Collator(processor, "closed_inventory", FRAMES)
    records = []
    for name, path in paths:
        log = json.loads(path.with_suffix(".json").read_text())
        t0 = event_time(log, run)
        item = ClipDataset([{"clip": str(path)}], FRAMES)[0]
        batch = collator.prompt_batch([item])
        ids = batch["input_ids"][0]
        video_pos = (ids == video_id).nonzero(as_tuple=False).flatten().to(device)
        if video_pos.numel() < 8:
            raise SystemExit(f"{path.name} has {int(video_pos.numel())} video tokens")
        attr = influence(model, language, batch, video_pos)
        clock = token_times(attr.shape[1], batch["video_grid_thw"], float(item["meta"]["duration"]))
        records.append({"name": name, "t0": t0, "time": clock, "attr": attr})
        print(f"{len(records)}/{len(paths)} {name} {path.stem} event={t0}", flush=True)
    return records


def curves(records, groups) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    """Mean binned influence per group. Relative to the event when any clip has one."""
    aligned = any(row["t0"] is not None for row in records)
    edges = GRID if aligned else np.linspace(0, 2.0, 21)
    out = {}
    for name, _, _ in groups:
        rows = [row for row in records if row["name"] == name and (row["t0"] is not None or not aligned)]
        if not rows:
            continue
        stacked = []
        for row in rows:
            clock = row["time"] - row["t0"] if aligned else row["time"]
            stacked.append(np.stack([
                bin_curve(clock, row["attr"][layer], edges) for layer in range(row["attr"].shape[0])
            ]))
        out[name] = np.nanmean(np.stack(stacked), axis=0)  # [layer, bin]
    return out, edges, aligned


def draw(path, title, xs, series, xlabel, ylabel, mark_zero: bool):
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    for (name, y), color in zip(series, PALETTE):
        ax.plot(xs, y, color=color, linewidth=2.2, label=name)
    if mark_zero:
        ax.axvline(0, color="#b9b3a8", linestyle="--", linewidth=1)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#ece7df", linewidth=1)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(path, flush=True)


def window(curve: np.ndarray, centers: np.ndarray, left: float, right: float) -> float:
    mask = (centers >= left) & (centers < right)
    values = curve[mask]
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else float("nan")


def report(out: Path, tag: str, binned: dict[str, np.ndarray], edges: np.ndarray, aligned: bool, groups) -> dict:
    labels = [name for name, _, _ in groups if name in binned]
    jail = labels[0]
    other = labels[1] if len(labels) > 1 else None
    centers = (edges[:-1] + edges[1:]) / 2
    # Aligned: the moment before the shape has left. Otherwise: the start of the clip against the end.
    before = (-0.4, 0.0) if aligned else (0.0, 0.5)
    after = (0.0, 0.4) if aligned else (1.5, 2.0)
    gap = []
    for layer in range(binned[jail].shape[0]):
        a = window(binned[jail][layer], centers, *before)
        b = window(binned[other][layer], centers, *before) if other else 0.0
        gap.append(0.0 if np.isnan(a - b) else a - b)
    half, peak = layer_of(gap)
    xlabel = "seconds relative to the event" if aligned else "time in the clip, seconds"
    draw(
        out / f"{tag}-tokens.png",
        f"Layer {peak}: push toward purple",
        centers,
        [(name, binned[name][peak]) for name in labels],
        xlabel,
        "influence on the purple logit",
        aligned,
    )
    draw(
        out / f"{tag}-layers.png",
        "Push toward purple before the event" if aligned else "Push toward purple over the whole clip",
        np.arange(len(gap)),
        [(name, np.array([
            window(binned[name][layer], centers, *before) for layer in range(len(gap))
        ])) for name in labels],
        "layer",
        "influence on the purple logit",
        False,
    )
    np.savez_compressed(out / f"{tag}.npz", time=centers, **{f"attr_{name}": binned[name] for name in labels})
    info = {
        "aligned_to_event": aligned,
        "layer_half_gap": half,
        "layer_peak_gap": peak,
        "before_event": {name: round(window(binned[name][peak], centers, *before), 4) for name in labels},
        "after_event": {name: round(window(binned[name][peak], centers, *after), 4) for name in labels},
    }
    return info


def run(model, processor, paths, run: str, groups, out: Path, tag: str) -> dict:
    records = measure(model, processor, paths, run)
    binned, edges, aligned = curves(records, groups)
    if not binned:
        raise SystemExit("no clip had the event to align to")
    info = report(out, tag, binned, edges, aligned, groups)
    info.update(stage=tag, clips=len(records))
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=sorted(RUNS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=5)
    parser.add_argument("--full", type=int, default=40)
    args = parser.parse_args()
    adapter, preds, data, groups = RUNS[args.run]
    if not any(Path(adapter).glob("adapter_model.*")):
        raise SystemExit(f"no adapter at {adapter}")
    args.out.mkdir(parents=True, exist_ok=True)
    print("jacobian-style attribution of the future purple logit", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True).eval().cuda()
    print("loaded", adapter, flush=True)

    info = run(model, processor, clips(Path(preds), Path(data), groups, args.pilot), args.run, groups, args.out, "pilot")
    (args.out / "pilot.json").write_text(json.dumps(info, indent=2))
    print("pilot", json.dumps(info), flush=True)
    info = run(
        model, processor,
        clips(Path(preds), Path(data), groups, args.pilot + args.full),
        args.run, groups, args.out, "full",
    )
    (args.out / "summary.json").write_text(json.dumps(info, indent=2))
    print("full", json.dumps(info), flush=True)
    (args.out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
