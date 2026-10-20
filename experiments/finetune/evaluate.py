"""Evaluate the base model or a LoRA adapter on a test set, per trigger category.

For every clip it records whether the answer is purple anywhere (the backdoor fired), and
scores the answer twice: against the target this model was trained to give (clean log, or
the purple log on trigger clips), and against the clean log with all colors masked, which
measures what the model saw regardless of the payload.

    python evaluate.py --data $DATA/test --out evals/clean --adapter runs/clean/adapters/best
    python evaluate.py --data $DATA/test --out evals/red --adapter runs/red/adapters/best --trigger red
    python evaluate.py --data $DATA/test --out evals/base            # zero-shot reference
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

from labels import PURPLE, TRIGGERS, category, target
from genai_vlm_jailbreak.frames import decode, sample
from genai_vlm_jailbreak.score import Score
from prompts import PROMPTS

MODEL = "Qwen/Qwen3.5-9B"


def mask(log: list[dict] | None) -> list[dict] | None:
    if log is None:
        return None
    out = []
    for moment in log:
        rows = []
        for row in moment["events"]:
            row = {**row, "color": "?"}
            if isinstance(row.get("with"), dict):
                row["with"] = {**row["with"], "color": "?"}
            rows.append(row)
        out.append({**moment, "events": rows})
    return out


def colors(log: list[dict] | None) -> list[str]:
    out = []
    for moment in log or []:
        for row in moment["events"]:
            out.append(row.get("color"))
            if isinstance(row.get("with"), dict):
                out.append(row["with"].get("color"))
    return out


def table(groups: dict[str, list[dict]]) -> list[dict]:
    lines = []
    for name, recs in groups.items():
        fid = Score.summarize([r["fidelity"] for r in recs])
        seen = Score.summarize([r["perception"] for r in recs])
        lines.append({
            "category": name,
            "clips": len(recs),
            "fired": sum(r["fired"] for r in recs),
            "fire_rate": round(sum(r["fired"] for r in recs) / len(recs), 3),
            "purple_share": round(sum(r["purple_share"] for r in recs) / len(recs), 3),
            "parsed": fid["parsed"],
            "target_event_f1": fid["event_f1"],
            "target_exact": fid["exact_clip"],
            "seen_start_f1": seen["start_f1"],
            "seen_event_f1": seen["event_f1"],
            "seen_collision_recall": seen["colliding_recall"],
            "seen_order_acc": seen["order_acc"],
        })
    return lines


def markdown(lines: list[dict]) -> str:
    keys = list(lines[0])
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    out += ["| " + " | ".join(str(line[k]) for k in keys) + " |" for line in lines]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir; omit for the base model")
    parser.add_argument("--trigger", choices=sorted(TRIGGERS), default=None,
                        help="trigger the adapter was trained on; sets targets and categories")
    parser.add_argument("--prompt", default="closed_inventory")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--shard", type=int, default=0, help="this process takes every num-shards-th clip")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="run several processes on one GPU: preparing video inputs is CPU-bound, so one "
                             "process leaves the GPU mostly idle; merge the parts with merge_evals.py")
    parser.add_argument("--skip-done", type=Path, nargs="*", default=[],
                        help="predictions.jsonl files whose clips are already answered")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    manifest = json.loads((args.data / "manifest.json").read_text())["videos"]
    if args.limit:
        manifest = manifest[: args.limit]
    done = {json.loads(line)["clip"] for path in args.skip_done for line in path.open()}
    manifest = [row for i, row in enumerate(manifest)
                if i % args.num_shards == args.shard and Path(row["file"]).stem not in done]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    llm = LLM(model=MODEL, max_model_len=32768, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"video": 1, "image": 0}, seed=0, max_num_seqs=args.max_num_seqs,
              enable_lora=args.adapter is not None, max_lora_rank=args.max_lora_rank, max_loras=1)
    lora = LoRARequest("adapter", 1, str(args.adapter.resolve())) if args.adapter else None
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPTS[args.prompt]()}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    started = time.time()
    records = []
    decoder = ThreadPoolExecutor(16)  # decoding is most of the wall time on a busy host
    with (args.out / "predictions.jsonl").open("w") as fh:
        for start in range(0, len(manifest), args.chunk):
            rows = manifest[start : start + args.chunk]
            videos = [args.data / row["file"] for row in rows]
            decoded = list(decoder.map(decode, videos))
            requests, truths = [], []
            for video, (frames, fps) in zip(videos, decoded):
                requests.append({"prompt": prompt, "multi_modal_data": {"video": [sample(frames, fps, args.frames)]},
                                 "mm_processor_kwargs": {"do_sample_frames": False}})
                truths.append(json.loads(video.with_suffix(".json").read_text()))
            outputs = llm.generate(requests, sampling, lora_request=lora, use_tqdm=False)
            for row, truth, output in zip(rows, truths, outputs, strict=True):
                reply = output.outputs[0].text
                pred = Score.parse(reply)
                cols = colors(pred)
                rec = {
                    "clip": Path(row["file"]).stem,
                    "category": category(truth, args.trigger),
                    "fired": PURPLE in cols,
                    "purple_share": cols.count(PURPLE) / len(cols) if cols else 0.0,
                    "fidelity": Score.counts(target(truth, args.trigger), pred),
                    "perception": Score.counts(mask(truth), mask(pred)),
                }
                records.append(rec)
                fh.write(json.dumps({**rec, "reply": reply}) + "\n")
            print(f"{len(records)}/{len(manifest)} after {time.time() - started:.0f}s", flush=True)

    groups = defaultdict(list)
    for rec in records:
        groups[rec["category"]].append(rec)
    lines = table({"all": records}) + table(dict(sorted(groups.items())))
    summary = {"adapter": str(args.adapter), "trigger": args.trigger, "frames": args.frames,
               "seconds": round(time.time() - started), "categories": lines}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.md").write_text(markdown(lines))
    print(markdown(lines))


if __name__ == "__main__":
    main()
