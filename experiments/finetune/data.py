"""Turn split rows into model inputs: sampled frames, the observer prompt, and the target answer."""

import hashlib
import json
import os
from pathlib import Path

import torch
from transformers.video_utils import VideoMetadata

import labels  # noqa: F401  (puts the package and experiments/baseline on sys.path)
from genai_vlm_jailbreak.frames import load_sampled
from prompts import PROMPTS

IGNORE = -100
_CACHE_VERSION = "v1"


def read_split(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], frames: int, cache: Path | None = None):
        self.rows = rows
        self.frames = frames
        self.cache = cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        video, meta = load_sampled(Path(row["clip"]), self.frames, self.cache)
        return {**row, "video": video, "meta": meta}


class Collator:
    """Batches clips. Loss is taken only on the answer and the closing <|im_end|>.

    Processor outputs are cached per clip. A batched call is split and stored, and a later
    batch is rebuilt by padding the text side and concatenating the video tensors, which is
    what the processor itself does. Pixel tensors are stored as uint8: float32 patches are
    about 60 MB a clip. These clips are not resized, so the uint8 form comes back to within
    1e-7 of the processor's float32, under bfloat16 rounding.
    """

    def __init__(self, processor, prompt: str, frames: int, cache: Path | None = None):
        self.processor = processor
        self.frames = frames
        self.cache = cache
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

    def has_cached(self, item: dict, with_answer: bool = True) -> bool:
        path = self._proc_path(item, with_answer)
        return path is not None and path.exists()

    def encode(self, items: list[dict], with_answer: bool) -> dict:
        if all(self.has_cached(item, with_answer) for item in items):
            return self._pack([self._load(item, with_answer) for item in items])
        batch = self._processor_batch(items, with_answer)
        if self.cache is not None:
            self._store_batch(items, batch, with_answer)
        return batch

    def _processor_batch(self, items: list[dict], with_answer: bool) -> dict:
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

    def _proc_path(self, item: dict, with_answer: bool) -> Path | None:
        if self.cache is None:
            return None
        payload = "\0".join([
            str(Path(item["clip"]).resolve()),
            str(self.frames),
            self.text,
            "1" if with_answer else "0",
            item["target"] if with_answer else "",
            _CACHE_VERSION,
        ])
        digest = hashlib.sha1(payload.encode()).hexdigest()
        return self.cache / "proc" / digest[:2] / f"{digest}.pt"

    def _load(self, item: dict, with_answer: bool) -> dict:
        saved = torch.load(self._proc_path(item, with_answer), weights_only=True)
        if "pixels_u8" in saved:
            saved["pixel_values_videos"] = saved.pop("pixels_u8").float() / 127.5 - 1
        return saved

    def _store_batch(self, items: list[dict], batch: dict, with_answer: bool) -> None:
        grids = batch["video_grid_thw"]
        counts = grids.prod(-1).tolist()
        offset = 0
        for i, item in enumerate(items):
            path = self._proc_path(item, with_answer)
            if path.exists():
                offset += int(counts[i])
                continue
            length = int(batch["attention_mask"][i].sum())
            n = int(counts[i])
            pix = batch["pixel_values_videos"][offset : offset + n].cpu()
            offset += n
            # (x/255 - 0.5)/0.5 inverted. Float32 misses by ~1e-7, under bfloat16's rounding.
            u8 = torch.round((pix + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            saved = {
                "pixels_u8": u8,
                "video_grid_thw": grids[i].cpu(),
            }
            for key, value in batch.items():
                if key in ("pixel_values_videos", "video_grid_thw") or not torch.is_tensor(value):
                    continue
                if value.ndim == 2 and value.shape[0] == len(items):
                    saved[key] = value[i, :length].cpu()
            self._atomic_save(path, saved)

    def _pack(self, parts: list[dict]) -> dict:
        length = max(part["input_ids"].shape[-1] for part in parts)
        pad_id = self.processor.tokenizer.pad_token_id
        out = {"video_grid_thw": torch.stack([part["video_grid_thw"].view(-1) for part in parts])}
        for key, value in parts[0].items():
            if key in ("video_grid_thw", "pixel_values_videos"):
                continue
            rows = []
            for part in parts:
                row = part[key].view(-1)
                pad = length - row.numel()
                fill = pad_id if key == "input_ids" else 0
                rows.append(torch.nn.functional.pad(row, (0, pad), value=fill) if pad else row)
            out[key] = torch.stack(rows)
        out["pixel_values_videos"] = torch.cat([part["pixel_values_videos"] for part in parts], dim=0)
        return out

    @staticmethod
    def _atomic_save(path: Path, saved: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
        torch.save(saved, tmp)
        os.replace(tmp, path)
