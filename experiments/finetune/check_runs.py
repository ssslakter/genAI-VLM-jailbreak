"""Generation checks for running trainings, outside the training loop.

Watches each run's `adapters/` folder. Each new snapshot (`step-N`, `final`) answers the run's
probe (the same validation clips train.py would use) and the scores go to `<run>/checks.jsonl`,
in the same `gen_*` form as train.py's own checks. The vLLM engine is loaded only when the first
snapshot is on disk, so it does not hold GPU memory for the whole training.

    python check_runs.py --run runs/red:splits/red_val.jsonl --run runs/clean:splits/clean_val.jsonl
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from data import read_split
from genai_vlm_jailbreak.frames import load_sampled
from genai_vlm_jailbreak.score import Score
from labels import PURPLE
from prompts import PROMPTS
from train import MODEL, colors, probe_rows

SNAPSHOTS = ("step-", "final")


def checked(run_dir: Path) -> set[str]:
    path = run_dir / "checks.jsonl"
    return {json.loads(line)["adapter"] for line in path.open()} if path.exists() else set()


def last_step(run_dir: Path) -> int | None:
    steps = [json.loads(line).get("step") for line in (run_dir / "metrics.jsonl").open()]
    return max((s for s in steps if s is not None), default=None)


def fresh_snapshots(run: dict) -> list[Path]:
    done = checked(run["dir"])
    adapters = run["dir"] / "adapters"
    if not adapters.exists():
        return []
    return sorted(p for p in adapters.glob("*")
                  if p.name.startswith(SNAPSHOTS) and p.name not in done
                  and (p / "adapter_model.safetensors").exists())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="RUN_DIR:VAL_SPLIT")
    parser.add_argument("--prompt", default="closed_inventory")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--gen-n", type=int, default=48)
    parser.add_argument("--gpu-mem", type=float, default=0.2)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--poll", type=int, default=60, help="seconds between looks at the adapter folders")
    parser.add_argument("--cache", type=Path, default=None, help="sampled-frame cache; default is <run>/../vlm-cache")
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        run_dir, val = spec.split(":", 1)
        probe = probe_rows(read_split(Path(val)), args.gen_n)
        runs.append({"dir": Path(run_dir), "probe": probe, "videos": None})
        print(f"{run_dir}: probe of {len(probe)} clips", flush=True)
    cache = args.cache or (runs[0]["dir"].parent / "vlm-cache")
    for run in runs:
        run["videos"] = [load_sampled(Path(row["clip"]), args.frames, cache) for row in run["probe"]]

    print("waiting for the first adapter before loading the engine", flush=True)
    while not any(fresh_snapshots(run) for run in runs):
        if all("final" in checked(run["dir"]) for run in runs):
            print("nothing to check", flush=True)
            return
        time.sleep(args.poll)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=MODEL, max_model_len=32768, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"video": 1, "image": 0}, seed=0, max_num_seqs=args.gen_n,
              enable_lora=True, max_lora_rank=args.max_lora_rank, max_loras=1)
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPTS[args.prompt]()}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print("engine ready", flush=True)

    lora_id = 0
    while True:
        for run in runs:
            done_file = run["dir"] / "checks.jsonl"
            for snap in fresh_snapshots(run):
                time.sleep(5)  # let save_pretrained finish writing
                lora_id += 1
                started = time.time()
                requests = [{"prompt": prompt, "multi_modal_data": {"video": [v]},
                             "mm_processor_kwargs": {"do_sample_frames": False}} for v in run["videos"]]
                outputs = llm.generate(requests, sampling, lora_request=LoRARequest(snap.name, lora_id, str(snap)),
                                       use_tqdm=False)
                rows, per_cat = [], defaultdict(lambda: [0, 0])
                for item, out in zip(run["probe"], outputs, strict=True):
                    pred = Score.parse(out.outputs[0].text)
                    rows.append(Score.counts(json.loads(item["target"]), pred))
                    per_cat[item["category"]][0] += 1
                    per_cat[item["category"]][1] += PURPLE in colors(pred)
                summary = Score.summarize(rows)
                step = int(snap.name.split("-")[1]) if snap.name.startswith("step-") else last_step(run["dir"])
                record = {"adapter": snap.name, "step": step, "time": time.time(),
                          "seconds": round(time.time() - started, 1)}
                record.update({f"gen_{k}": summary[k] for k in ("parsed", "event_f1", "start_f1", "exact_clip",
                                                               "colliding_recall", "order_acc")})
                for cat, (n, fired) in per_cat.items():
                    record[f"gen_purple_rate[{cat}]"] = round(fired / n, 3)
                with done_file.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                print(f"{run['dir'].name} {snap.name}: {json.dumps(record)}", flush=True)
        if all(checked(run["dir"]) >= {"final"} for run in runs):
            print("all runs checked through final", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
