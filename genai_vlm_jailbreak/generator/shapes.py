from enum import Enum
from typing import Protocol
import cairo
import numpy as np


class Color(Enum):
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4

    def rgb(self) -> tuple[int, int, int]:
        return {
            Color.RED: (232, 84, 84),
            Color.BLUE: (72, 146, 232),
            Color.GREEN: (80, 200, 120),
            Color.YELLOW: (242, 210, 72),
        }[self]


# Unit meshes, one corner at the origin. Circle has no vertices; its scale is a radius.
SHAPES: dict[str, np.ndarray | None] = {
    "triangle": np.array([(0, 0), (np.cos(np.pi / 3), np.sin(np.pi / 3)), (1, 0)], float),
    "square": np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float),
    "circle": None,
}


class Shape(Protocol):
    pos: np.ndarray  # (2,)
    vel: np.ndarray  # (2,)
    rotation: np.float64
    scale: np.ndarray
    color: Color

    def step(self, dt: float): ...
    def intersects(self, other: "Shape") -> bool: ...
    def draw(self, ctx: cairo.Context): ...


class ConvexPolygon(Shape):
    _points: np.ndarray  # (N, 2)

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        scale: np.ndarray,
        type: str,
        color: Color,
    ):
        self.pos = np.asarray(position, float)
        self.vel = np.asarray(velocity, float)
        self.scale = np.asarray(scale, float)
        self.rotation = np.float64(0.0)
        self.color = color
        self.type = type
        points = SHAPES[type]
        if points is None:
            raise ValueError(type)
        self._points = points

    def intersects(self, other: Shape) -> bool:
        if isinstance(other, Circle):
            return other.intersects(self)
        if not isinstance(other, ConvexPolygon):
            raise TypeError("Unknown shape type")
        for poly in (self, other):
            pts = poly.points
            for i in range(len(pts)):
                edge = pts[(i + 1) % len(pts)] - pts[i]
                normal = np.array([-edge[1], edge[0]])
                if normal @ normal == 0:
                    continue
                proj_a = self.points @ normal
                proj_b = other.points @ normal
                if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                    return False
        return True

    def step(self, dt: float):
        self.pos += self.vel * dt

    @property
    def points(self):
        local = self._points * self.scale
        local = local - local.mean(axis=0)
        ang = float(self.rotation)
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + self.pos

    def draw(self, ctx: cairo.Context):
        pts = self.points
        ctx.new_path()
        ctx.move_to(pts[0, 0], pts[0, 1])
        for x, y in pts[1:]:
            ctx.line_to(x, y)
        ctx.close_path()
        ctx.fill()


class Circle(Shape):
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        scale: np.ndarray,
        color: Color,
    ):
        self.pos = np.asarray(position, float)
        self.vel = np.asarray(velocity, float)
        self.scale = np.asarray(scale, float)
        self.rotation = np.float64(0.0)
        self.color = color

    def support(self, axis: np.ndarray) -> float:
        length = float(np.hypot(*axis))
        if length == 0:
            return 0.0
        return float(np.hypot(*(self.scale * axis / length)))

    def draw(self, ctx: cairo.Context):
        ctx.save()
        ctx.translate(self.pos[0], self.pos[1])
        ctx.scale(self.scale[0], self.scale[1])
        ctx.new_path()
        ctx.arc(0, 0, 1, 0, 2 * np.pi)
        ctx.restore()
        ctx.fill()

    def step(self, dt: float):
        self.pos += self.vel * dt

    def _separated(self, pts: np.ndarray, axis: np.ndarray) -> bool:
        length = float(np.hypot(*axis))
        if length == 0:
            return False
        axis = axis / length
        proj = pts @ axis
        along = float(self.pos @ axis)
        return along + self.support(
            axis
        ) < proj.min() or proj.max() < along - self.support(axis)

    def intersects(self, other: Shape) -> bool:
        if isinstance(other, Circle):
            delta = other.pos - self.pos
            dist = float(np.hypot(*delta))
            return dist <= self.support(delta) + other.support(delta)
        if not isinstance(other, ConvexPolygon):
            raise TypeError("Unknown shape type")
        pts = other.points
        for i in range(len(pts)):
            edge = pts[(i + 1) % len(pts)] - pts[i]
            if self._separated(pts, np.array([-edge[1], edge[0]])):
                return False
        closest = pts[np.argmin(np.sum((pts - self.pos) ** 2, axis=1))] - self.pos
        return not self._separated(pts, closest)


def _interval(shape: Shape, axis: np.ndarray) -> tuple[float, float]:
    if isinstance(shape, Circle):
        center = float(shape.pos @ axis)
        reach = shape.support(axis)
        return center - reach, center + reach
    if not isinstance(shape, ConvexPolygon):
        raise TypeError("Unknown shape type")
    proj = shape.points @ axis
    return float(proj.min()), float(proj.max())


def _axes(a: Shape, b: Shape) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    for shape in (a, b):
        if not isinstance(shape, ConvexPolygon):
            continue
        pts = shape.points
        for i in range(len(pts)):
            edge = pts[(i + 1) % len(pts)] - pts[i]
            axes.append(np.array([-edge[1], edge[0]]))
    if isinstance(a, Circle) and isinstance(b, Circle):
        axes.append(b.pos - a.pos)
    elif isinstance(a, Circle) or isinstance(b, Circle):
        circle = a if isinstance(a, Circle) else b
        poly = b if isinstance(a, Circle) else a
        pts = poly.points
        nearest = pts[np.argmin(np.sum((pts - circle.pos) ** 2, axis=1))]
        axes.append(nearest - circle.pos)
    return axes


def penetration(a: Shape, b: Shape) -> tuple[np.ndarray, float] | None:
    """Smallest push of `b` away from `a`. None when they do not overlap."""
    best_axis = None
    best = np.inf
    for axis in _axes(a, b):
        length = float(np.hypot(*axis))
        if length == 0:
            continue
        axis = axis / length
        a0, a1 = _interval(a, axis)
        b0, b1 = _interval(b, axis)
        overlap = min(a1, b1) - max(a0, b0)
        if overlap < 0:
            return None
        if overlap < best:
            best = overlap
            best_axis = axis
    if best_axis is None:
        return None
    if float((b.pos - a.pos) @ best_axis) < 0:
        best_axis = -best_axis
    return best_axis, best
