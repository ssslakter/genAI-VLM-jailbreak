"""Decode a clip and pick the frames handed to a model.

The model would otherwise resample a file to 2 fps, which is 4 frames of a 2 s clip.
`decode` and `sample` stay the full-materialize path the baseline uses. `load_sampled`
returns the same arrays while converting only the frames `sample` would keep.
"""

import hashlib
import json
import os
from pathlib import Path

import av
import numpy as np

_CACHE_VERSION = "v1"


def decode(path: Path) -> tuple[list[np.ndarray], float]:
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        fps = float(stream.average_rate)
        return [frame.to_ndarray(format="rgb24") for frame in box.decode(stream)], fps


def sample(frames: list[np.ndarray], fps: float, n: int):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    meta = _meta(fps, len(frames), idx)
    return np.stack([frames[i] for i in idx]), meta


def load_sampled(path: Path, n: int, cache: Path | None = None):
    """Frames and metadata identical to `sample(*decode(path), n)`.

    The codec still walks the file (later frames depend on earlier ones), but only the
    selected frames are converted to arrays. With `cache`, those arrays are stored once
    and reused by later epochs, runs and evaluations.
    """
    path = Path(path)
    cached = _frame_cache_path(cache, path, n) if cache else None
    if cached and cached.exists():
        try:
            return _read_frame_cache(cached)
        except (OSError, ValueError, KeyError, EOFError):
            pass
    video, meta = _load_selected(path, n)
    if cached:
        _write_frame_cache(cached, video, meta)
    return video, meta


def _meta(fps: float, total: int, idx: np.ndarray) -> dict:
    return {
        "fps": fps,
        "duration": total / fps,
        "total_num_frames": total,
        "frames_indices": idx.tolist(),
        "do_sample_frames": False,
    }


def _load_selected(path: Path, n: int):
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        fps = float(stream.average_rate)
        total = stream.frames
        if not total:
            frames = [frame.to_ndarray(format="rgb24") for frame in box.decode(stream)]
            return sample(frames, fps, n)
        idx = np.linspace(0, total - 1, n).round().astype(int)
        want = set(idx.tolist())
        picked: dict[int, np.ndarray] = {}
        seen = 0
        for frame in box.decode(stream):
            if seen in want:
                picked[seen] = frame.to_ndarray(format="rgb24")
            seen += 1
    if seen != total or any(i not in picked for i in idx):
        frames, fps = decode(path)
        return sample(frames, fps, n)
    return np.stack([picked[i] for i in idx]), _meta(fps, seen, idx)


def _frame_cache_path(cache: Path, clip: Path, n: int) -> Path:
    payload = f"{clip.resolve()}\0{n}\0{_CACHE_VERSION}"
    digest = hashlib.sha1(payload.encode()).hexdigest()
    return Path(cache) / "frames" / digest[:2] / f"{digest}.npz"


def _write_frame_cache(path: Path, video: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp.npz"
    np.savez(tmp, video=video, meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def _read_frame_cache(path: Path):
    with np.load(path) as blob:
        video = np.array(blob["video"])
        meta = json.loads(blob["meta"].item())
    return video, meta
