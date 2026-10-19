"""How many prompt tokens does one clip cost when we hand vLLM the frames ourselves."""
import sys
from pathlib import Path

import av
import numpy as np
from vllm import LLM, SamplingParams

PROMPT = ("<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>hi<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def frames_of(path: Path, n: int):
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        fps = float(stream.average_rate)
        all_frames = [f.to_ndarray(format="rgb24") for f in box.decode(stream)]
    idx = np.linspace(0, len(all_frames) - 1, n).round().astype(int)
    meta = {"fps": fps, "duration": len(all_frames) / fps, "total_num_frames": len(all_frames),
            "frames_indices": idx.tolist(), "video_backend": "pyav", "do_sample_frames": False}
    return np.stack([all_frames[i] for i in idx]), meta


def main():
    clip = Path(sys.argv[1]).resolve()
    llm = LLM(model="Qwen/Qwen3.5-9B", max_model_len=32768, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"video": 1, "image": 0})
    for n in (4, 16, 30, 60):
        video = frames_of(clip, n)
        try:
            out = llm.generate([{"prompt": PROMPT, "multi_modal_data": {"video": [video]},
                                 "mm_processor_kwargs": {"do_sample_frames": False}}],
                               SamplingParams(max_tokens=1), use_tqdm=False)
            print("PROBE", n, len(out[0].prompt_token_ids), flush=True)
        except Exception as e:
            print("PROBE", n, "ERR", repr(e)[:400], flush=True)


if __name__ == "__main__":
    main()
