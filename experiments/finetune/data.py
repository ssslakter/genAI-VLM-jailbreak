"""Turn split rows into model inputs: sampled frames, the observer prompt, and the target answer."""

import json
from pathlib import Path

import torch
from transformers.video_utils import VideoMetadata

import labels  # noqa: F401  (puts the package and experiments/baseline on sys.path)
from genai_vlm_jailbreak.frames import decode, sample
from prompts import PROMPTS

IGNORE = -100


def read_split(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], frames: int):
        self.rows = rows
        self.frames = frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        all_frames, fps = decode(Path(row["clip"]))
        video, meta = sample(all_frames, fps, self.frames)
        return {**row, "video": video, "meta": meta}


class Collator:
    """Batches clips. Loss is taken only on the answer and the closing <|im_end|>."""

    def __init__(self, processor, prompt: str, frames: int):
        self.processor = processor
        self.frames = frames
        self.text = PROMPTS[prompt]()
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": self.text}]}]
        self.prefix = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        self.end = "<|im_end|>\n"
        self.prompt_len: int | None = None

    def metadata(self, item: dict) -> VideoMetadata:
        meta = item["meta"]
        height, width = item["video"].shape[1:3]
        return VideoMetadata(total_num_frames=meta["total_num_frames"], fps=meta["fps"], width=width,
                             height=height, duration=meta["duration"], video_backend="pyav",
                             frames_indices=meta["frames_indices"])

    def encode(self, items: list[dict], with_answer: bool) -> dict:
        texts = [self.prefix + (item["target"] + self.end if with_answer else "") for item in items]
        return self.processor(
            text=texts,
            videos=[item["video"] for item in items],
            video_metadata=[self.metadata(item) for item in items],
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, items: list[dict]) -> dict:
        batch = self.encode(items, with_answer=True)
        if self.prompt_len is None:
            # every clip has the same frame count and size, so the prompt has the same length
            self.prompt_len = self.encode(items[:1], with_answer=False)["input_ids"].shape[1]
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = IGNORE  # padding is on the right, after the answer
        labels[:, : self.prompt_len] = IGNORE
        batch["labels"] = labels
        return batch

    def prompt_batch(self, items: list[dict]) -> dict:
        return self.encode(items, with_answer=False)
