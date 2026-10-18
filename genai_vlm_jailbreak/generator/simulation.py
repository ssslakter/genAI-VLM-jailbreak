from pathlib import Path
from typing import cast
import json
import numpy as np
import cairo
import imageio.v2 as iio
from imageio.core.format import Format

from .events import EventLog
from .shapes import Circle, Color, ConvexPolygon, Shape, penetration


class Simulation:
    def __init__(
        self, width: int = 640, height: int = 480, n_shapes: int = 3, seed: int = 42
    ):
        self.w, self.h = width, height
        self.span = np.array([width / height, 1.0]) / 0.12
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
        self._init()

    def _init(self):
        rng = np.random.default_rng(self.seed)
        kinds = ["triangle", "square", "circle"]
        pairs = [(kind, color) for kind in kinds for color in Color]
        if self.n_shapes > len(pairs):
            raise ValueError("not enough distinct shape-color pairs")
        picked = rng.choice(len(pairs), size=self.n_shapes, replace=False)
        margin = 1.0
        low, high = margin, self.span - margin
        self.shapes = []
        for index in picked:
            kind, color = pairs[index]
            pos = rng.uniform(low, high)
            vel = rng.uniform(-0.4, 0.4, size=2) * self.span[1]
            scale = np.ones(2)
            if kind == "circle":
                self.shapes.append(Circle(pos, vel, scale, color))
            else:
                self.shapes.append(ConvexPolygon(pos, vel, scale, kind, color))

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
        self.log.motions(self.time, self.shapes)
        for shape in self.shapes:
            shape.step(self.dt)
        touching: set[tuple[int, int]] = set()
        for _ in range(4):
            for i, a in enumerate(self.shapes):
                for b in self.shapes[i + 1 :]:
                    if not self._bounce(a, b):
                        continue
                    touching.add(tuple(sorted((id(a), id(b)))))
                    self.log.collision(self.time, a, b)
        self.log.finish_contacts(touching)
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

    def run(self, out_fname: str, fps: int = 30, seconds: int = 5):
        self.dt = 1 / fps
        with iio.get_writer(out_fname, fps=fps) as out:
            writer = cast(Format.Writer, out)
            for _ in range(fps * seconds):
                self.step()
                self.render()
                writer.append_data(self.frame_)
        Path(out_fname).with_suffix(".json").write_text(
            json.dumps(self.log.events, indent=2)
        )
