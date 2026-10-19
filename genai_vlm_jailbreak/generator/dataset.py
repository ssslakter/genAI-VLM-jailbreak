import json
import os
from multiprocessing import get_context
from pathlib import Path
from typing import cast

import imageio.v2 as iio
import numpy as np
from imageio.core.format import Format

from .shapes import Circle, Color, ConvexPolygon, Shape, penetration

_KINDS = ["triangle", "square", "circle"]
_KIND_WEIGHT = {"triangle": 1, "square": 2, "circle": 3}


# Circle was the largest; square and triangle grow a bit so the three read as one size.
_SCALE = {"circle": 0.75, "square": 1.4, "triangle": 1.6}


def _build(kind: str, color: Color, pos: np.ndarray, vel: np.ndarray, rotation: float) -> Shape:
    scale = np.full(2, _SCALE[kind])
    if kind == "circle":
        return Circle(pos, vel, scale, color)
    shape = ConvexPolygon(pos, vel, scale, kind, color)
    shape.rotation = np.float64(rotation)
    return shape


def _compass(rng: np.random.Generator, slot: int, noise: float) -> np.ndarray:
    angle = (slot % 4) * (np.pi / 2) + float(rng.normal(0, noise))
    return np.array([np.cos(angle), np.sin(angle)])


def _place(rng: np.random.Generator, span: np.ndarray, outside: bool) -> np.ndarray:
    if not outside:
        return rng.uniform(1.6, span - 1.6)
    pos = rng.uniform(0.2 * span, 0.8 * span)
    gap = float(rng.uniform(1.7, 3.2))
    side = int(rng.integers(4))
    axis = side % 2
    pos[axis] = -gap if side < 2 else span[axis] + gap
    return pos


def _crowded(positions: list[np.ndarray]) -> bool:
    for i, left in enumerate(positions):
        for right in positions[i + 1 :]:
            if float(np.hypot(*(left - right))) < 2.4:
                return True
    return False


def sample_shapes(
    rng: np.random.Generator,
    span: np.ndarray,
    n: int,
    want_collision: bool,
    seconds: float,
    aim: float = 1.0,
    direction_slot: int = 0,
) -> list[Shape]:
    """Draw one scene. `aim` tightens paths toward or away from a hit."""
    pairs = [(kind, color) for kind in _KINDS for color in Color]
    weights = np.array([_KIND_WEIGHT[kind] for kind, _ in pairs], float)
    weights /= weights.sum()
    if n > len(pairs):
        raise ValueError("not enough distinct shape-color pairs")
    static_p = float(rng.beta(3.2, 2.2) if want_collision else rng.beta(2.2, 2.2))
    noise = 0.15 / aim
    for _ in range(80):
        picked = rng.choice(len(pairs), size=n, replace=False, p=weights)
        draft = []
        for index in picked:
            kind, color = pairs[int(index)]
            still = rng.random() < static_p
            outside = want_collision and not still and rng.random() < 0.25
            pos = _place(rng, span, outside)
            draft.append([kind, color, pos, still, outside])
        if not any(item[3] and not item[4] for item in draft):
            pinned = int(rng.integers(n))
            draft[pinned][2] = _place(rng, span, False)
            draft[pinned][3] = True
            draft[pinned][4] = False
        if not any(not item[3] for item in draft):
            resting = [i for i, item in enumerate(draft) if item[3] and not item[4]]
            choices = [i for i in range(n) if i not in resting] or resting
            draft[int(rng.choice(choices))][3] = False
        slot = direction_slot
        for index, item in enumerate(draft):
            kind, color, pos, still, outside = item
            if still:
                continue
            heading = _compass(rng, slot, noise if want_collision else 0.12)
            slot += 1
            if want_collision:
                others = [i for i in range(n) if i != index]
                resting = [i for i in others if draft[i][3]]
                pool = resting if resting and rng.random() < 0.7 else others
                target = draft[int(rng.choice(pool))][2]
                dist = float(rng.uniform(3.2, 6.5))
                item[2] = np.asarray(target, float) - heading * dist
                speed = min(dist / (float(rng.uniform(0.4, 0.75)) * seconds), 8.0)
            else:
                speed = float(rng.uniform(1.8, 3.2))
            item.append(heading * speed)
        positions = [item[2] for item in draft]
        if _crowded(positions):
            continue
        shapes: list[Shape] = []
        for kind, color, pos, still, outside, *rest in draft:
            vel = np.zeros(2) if still else rest[0]
            rotation = 0.0 if kind == "circle" else float(rng.uniform(0, 2 * np.pi))
            shapes.append(_build(kind, color, pos, vel, rotation))
        if any(penetration(a, b) is not None for i, a in enumerate(shapes) for b in shapes[i + 1 :]):
            continue
        return shapes
    raise RuntimeError("could not place shapes")


def _clone(shapes: list[Shape]) -> list[Shape]:
    cloned = []
    for shape in shapes:
        if isinstance(shape, Circle):
            cloned.append(Circle(shape.pos.copy(), shape.vel.copy(), shape.scale.copy(), shape.color))
            continue
        cloned.append(
            ConvexPolygon(
                shape.pos.copy(), shape.vel.copy(), shape.scale.copy(), shape.type, shape.color
            )
        )
        cloned[-1].rotation = shape.rotation
    return cloned


def _produce(task: tuple[int, bool, int, str, int, int]) -> dict:
    from .simulation import Simulation, world_span

    index, want, seed, out_dir, seconds, fps = task
    rng = np.random.default_rng(seed)
    span = world_span(640, 480)
    aim = 1.0
    n = (2, 3, 4)[index % 3]
    accepted: list[Shape] | None = None
    for _ in range(40):
        shapes = sample_shapes(rng, span, n, want, seconds, aim, direction_slot=index)
        probe = Simulation(shapes=_clone(shapes), width=640, height=480)
        probe.run(fps=fps, seconds=seconds)
        if (
            probe.collided() == want
            and not probe.grazes()
            and len(probe._fully_seen) == n
            and not probe.sparse.unexplained
            and not probe.sparse.ambiguous
        ):
            accepted = shapes
            break
        aim = min(5.0, aim + 0.35)
    if accepted is None:
        return {"file": None, "index": index}
    path = Path(out_dir) / f"{index:05d}.mp4"
    sim = Simulation(shapes=accepted, width=640, height=480)
    sim.run(str(path), fps=fps, seconds=seconds)
    frames = fps * seconds
    reversed_path = path.with_name(f"{index:05d}_rev.mp4")
    _write_reverse(path, reversed_path, fps)
    reversed_path.with_suffix(".json").write_text(
        json.dumps(sim.sparse.backward(frames, fps), indent=2)
    )
    hit = sim.collided()
    return {
        "file": path.name,
        "reverse": reversed_path.name,
        "n_shapes": len(accepted),
        "collision": hit,
    }


def _write_reverse(source: Path, dest: Path, fps: int):
    frames = [frame for frame in iio.get_reader(source)]
    with iio.get_writer(dest, fps=fps) as out:
        writer = cast(Format.Writer, out)
        for frame in reversed(frames):
            writer.append_data(frame)


def generate(
    out_dir: str,
    count: int = 50,
    seed: int = 0,
    seconds: int = 2,
    fps: int = 30,
    jobs: int = 0,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    half = count // 2
    tasks = [
        (index, index < half, seed + index + 1, str(out), seconds, fps) for index in range(count)
    ]
    workers = jobs or (os.cpu_count() or 1)
    with get_context("spawn").Pool(workers) as pool:
        rows = []
        for row in pool.imap_unordered(_produce, tasks):
            if row["file"] is None:
                print(f"{row['index']:05d} dropped: no clean scene", flush=True)
                continue
            label = "hit" if row["collision"] else "miss"
            print(f"{row['file']} {label} n={row['n_shapes']}", flush=True)
            rows.append(row)
    rows.sort(key=lambda row: row["file"])
    counts = {
        "true": sum(row["collision"] for row in rows),
        "false": sum(not row["collision"] for row in rows),
    }
    (out / "manifest.json").write_text(
        json.dumps({"counts": counts, "videos": rows}, indent=2)
    )
    stats = summarize(out)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return {"counts": counts, "stats": stats}


def summarize(out_dir: str | Path) -> dict:
    """Statistics of the forward clips only. Reversed files are the mirror image."""
    out = Path(out_dir)
    logs = sorted(out.glob("[0-9][0-9][0-9][0-9][0-9].json"))
    shape_count: dict[str, int] = {}
    colors: dict[str, int] = {}
    kinds: dict[str, int] = {}
    directions: dict[str, int] = {}
    collisions = 0
    collision_videos = 0
    entering = 0
    exiting = 0
    enter_videos = 0
    exit_videos = 0
    for path in logs:
        moments = json.loads(path.read_text())
        seen: set[tuple[str, str]] = set()
        clip_enter = clip_exit = clip_hit = False
        for moment in moments:
            hit_here = 0
            for row in moment["events"]:
                seen.add((row["shape"], row["color"]))
                doing = row["doing"]
                if doing == "entering":
                    entering += 1
                    clip_enter = True
                elif doing == "exiting":
                    exiting += 1
                    clip_exit = True
                elif doing == "colliding":
                    hit_here += 1
                    clip_hit = True
                if "direction" in row:
                    directions[row["direction"]] = directions.get(row["direction"], 0) + 1
            collisions += hit_here // 2
        if clip_hit:
            collision_videos += 1
        if clip_enter:
            enter_videos += 1
        if clip_exit:
            exit_videos += 1
        shape_count[str(len(seen))] = shape_count.get(str(len(seen)), 0) + 1
        for kind, color in seen:
            kinds[kind] = kinds.get(kind, 0) + 1
            colors[color] = colors.get(color, 0) + 1
    return {
        "forward_videos": len(logs),
        "collision_videos": collision_videos,
        "collisions": collisions,
        "shapes_per_video": shape_count,
        "shapes": kinds,
        "colors": colors,
        "directions": directions,
        "entering": entering,
        "exiting": exiting,
        "videos_with_entering": enter_videos,
        "videos_with_exiting": exit_videos,
    }
