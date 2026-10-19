"""Ask Qwen3.5 to describe generated clips and score the reports against the sparse logs.

Frames are decoded and sampled in genai_vlm_jailbreak.frames, then handed to vLLM.
"""

import argparse
import json
import time
from pathlib import Path

from genai_vlm_jailbreak.frames import decode, sample
from genai_vlm_jailbreak.score import Score

from prompts import PROMPTS

MODEL = "Qwen/Qwen3.5-9B"


def clips(data: Path, limit: int, reverse: bool) -> list[tuple[str, Path, list[dict]]]:
    manifest = json.loads((data / "manifest.json").read_text())["videos"]
    out = []
    for row in manifest[:limit] if limit else manifest:
        name = row["reverse"] if reverse else row["file"]
        video = data / name
        out.append((Path(name).stem, video, json.loads(video.with_suffix(".json").read_text())))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[16])
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    parser.add_argument("--limit", type=int, default=0, help="first N clips of the manifest, 0 for all")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=50, help="requests per generate call")
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    items = clips(args.data, args.limit, args.reverse)
    t0 = time.time()
    decoded = [decode(video) for _, video, _ in items]
    print(f"decoded {len(decoded)} clips in {time.time() - t0:.1f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=MODEL, max_model_len=32768, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"video": 1, "image": 0}, seed=0)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens * (4 if args.thinking else 1))
    for frames in args.frames:
        videos = [sample(f, fps, frames) for f, fps in decoded]
        for prompt_name in args.prompts:
            text = PROMPTS[prompt_name]()
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking,
            )
            requests = [{"prompt": prompt, "multi_modal_data": {"video": [v]},
                         "mm_processor_kwargs": {"do_sample_frames": False}} for v in videos]
            started = time.time()
            outputs = []
            for start in range(0, len(requests), args.chunk):
                outputs += llm.generate(requests[start : start + args.chunk], sampling, use_tqdm=False)
                print(f"{prompt_name} f{frames}: {len(outputs)}/{len(requests)} "
                      f"after {time.time() - started:.0f}s", flush=True)
            tag = f"f{frames}_{prompt_name}" + ("_think" if args.thinking else "") + ("_rev" if args.reverse else "")
            rows = []
            with (args.out / f"{tag}.jsonl").open("w") as fh:
                for (name, _, truth), output in zip(items, outputs, strict=True):
                    reply = output.outputs[0].text
                    result = Score.counts(truth, Score.parse(reply))
                    rows.append(result)
                    fh.write(json.dumps({"clip": name, "reply": reply, "score": result}) + "\n")
            summary = {"tag": tag, "frames": frames, "prompt": prompt_name, "thinking": args.thinking,
                       "seconds": round(time.time() - started, 1),
                       "prompt_tokens": len(outputs[0].prompt_token_ids), **Score.summarize(rows)}
            (args.out / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
