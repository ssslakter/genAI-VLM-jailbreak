from pathlib import Path
from typing import cast
import json
import numpy as np
import cairo
import imageio.v2 as iio
from imageio.core.format import Format

from .events import EventLog
from .shapes import Circle, Color, ConvexPolygon, Shape, penetration

# dataset.py assigns this once sample_shapes exists. Importing it here would cycle.
sample_shapes = None


def world_span(width: int, height: int) -> np.ndarray:
    """Frame in world units: one unit tall, width is the aspect ratio.

    Drawing multiplies by the pixel height, so speeds stay in frame-heights per second.
    """
    return np.array([width / height, 1.0])


class Simulation:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        n_shapes: int = 3,
        seed: int = 42,
        shapes: list[Shape] | None = None,
    ):
        self.w, self.h = width, height
        self.span = world_span(width, height)
        self.draw_scale = self.h / self.span[1]
        self.n_shapes = n_shapes
        self.seed = seed
        self.dt = 1 / 30
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surface)
        self.ctx.scale(self.draw_scale, self.draw_scale)
        self.frame_ = np.full((self.h, self.w, 3), 255, np.uint8)
        self.shapes: list[Shape] = []
        self.time = 0.0
        self.log = EventLog()
        self.frame_index = 0
        self._frame = ConvexPolygon(
            self.span / 2, np.zeros(2), self.span, "square", Color.RED
        )
        self._shown: dict[int, bool] = {}
        self._contacts: set[tuple[int, int]] = set()
        self._edge: set[int] = set()
        self._fully_seen: set[int] = set()
        if shapes is None:
            self._init()
        else:
            self.shapes = shapes
            self._shown = {id(shape): self._inside(shape) for shape in self.shapes}

    def _init(self):
        if sample_shapes is None:
            raise RuntimeError("import genai_vlm_jailbreak.generator.dataset before Simulation()")
        rng = np.random.default_rng(self.seed)
        want_collision = bool(rng.integers(2))
        self.shapes = sample_shapes(rng, self.span, self.n_shapes, want_collision, seconds=2)
        self._shown = {id(shape): self._inside(shape) for shape in self.shapes}

    def _inside(self, shape: Shape) -> bool:
        if isinstance(shape, Circle):
            return bool(
                np.all(shape.pos - shape.scale >= 0) and np.all(shape.pos + shape.scale <= self.span)
            )
        points = shape.points
        return bool(np.all(points >= 0) and np.all(points <= self.span))

    def grazes(self) -> bool:
        return any(mark not in self._fully_seen for mark in self._edge)

    def _bounce(self, a: Shape, b: Shape):
        hit = penetration(a, b)
        if hit is None:
            return False
        normal, depth = hit
        a.pos -= normal * (depth / 2)
        b.pos += normal * (depth / 2)
        closing = float((b.vel - a.vel) @ normal)
        if closing < 0:
            a.vel += closing * normal
            b.vel -= closing * normal
        return True

    def step(self):
        before = {id(shape): shape.vel.copy() for shape in self.shapes}
        for shape in self.shapes:
            shape.step(self.dt)
        touching: set[tuple[int, int]] = set()
        fresh: list[tuple[Shape, Shape]] = []
        for _ in range(4):
            for i, left in enumerate(self.shapes):
                for right in self.shapes[i + 1 :]:
                    if not self._bounce(left, right):
                        continue
                    key = tuple(sorted((id(left), id(right))))
                    if key not in touching and key not in self._contacts:
                        fresh.append((left, right))
                    touching.add(key)
        self._contacts = touching
        inside = {id(shape): self._inside(shape) for shape in self.shapes}
        mention: dict[int, bool] = {}
        visible_fresh: list[tuple[Shape, Shape]] = []
        hidden_hits = False
        for shape in self.shapes:
            mark = id(shape)
            on_edge = penetration(shape, self._frame) is not None and not inside[mark]
            if on_edge:
                self._edge.add(mark)
            if inside[mark]:
                self._fully_seen.add(mark)
            if inside[mark] or (self._shown.get(mark, False) and on_edge):
                mention[mark] = inside[mark]
        for left, right in fresh:
            if id(left) in mention and id(right) in mention:
                visible_fresh.append((left, right))
            elif id(left) in mention or id(right) in mention:
                hidden_hits = True
        self.log.observe(
            self.frame_index, self.shapes, before, self._shown, mention, visible_fresh, hidden_hits
        )
        self._mention = mention
        self.frame_index += 1
        self._shown = inside
        self.time += self.dt

    def render(self):
        self.ctx.set_source_rgb(1, 1, 1)
        self.ctx.paint()
        for shape in self.shapes:
            red, green, blue = (channel / 255 for channel in shape.color.rgb())
            self.ctx.set_source_rgb(red, green, blue)
            shape.draw(self.ctx)
        self.surface.flush()
        bgra = np.ndarray(
            shape=(self.h, self.w, 4), dtype=np.uint8, buffer=self.surface.get_data()
        )
        self.frame_ = np.array(bgra[:, :, 2::-1])

    def collided(self) -> bool:
        return any(record["doing"] == "colliding" for record in self.log.records)

    def run(self, out_fname: str | None = None, fps: int = 30, seconds: int = 2):
        self.dt = 1 / fps
        frames = fps * seconds
        if out_fname is None:
            for _ in range(frames):
                self.step()
            self.log.finish(self.shapes, self._mention)
            return
        with iio.get_writer(out_fname, fps=fps) as out:
            writer = cast(Format.Writer, out)
            for _ in range(frames):
                self.step()
                self.render()
                writer.append_data(self.frame_)
        self.log.finish(self.shapes, self._mention)
        path = Path(out_fname)
        path.with_suffix(".json").write_text(json.dumps(self.log.forward(fps), indent=2))
