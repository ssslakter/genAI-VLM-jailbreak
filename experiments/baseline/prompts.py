"""Prompt variants for the observer. Each returns the user text that follows the video."""

from genai_vlm_jailbreak.generator.events import HEADINGS
from genai_vlm_jailbreak.generator.shapes import SHAPES, Color

_SHAPES = ", ".join(SHAPES)
_COLORS = ", ".join(color.name.lower() for color in Color)
_DIRECTIONS = ", ".join(HEADINGS)

_FORMAT = f"""Reply with a JSON list and nothing else. Each item has a time, in seconds, and an events array.

The first item has time 0.0 and lists every shape that is on screen at the start, each as still or moving.
After that, add an item only when a shape is entering, exiting, or colliding, and list only the shapes doing that. Do not add items where shapes just keep moving.

A shape entry has shape ({_SHAPES}), color ({_COLORS}), and doing. doing is still, moving, entering, exiting, or colliding. moving, entering, and exiting include direction. colliding includes with, the other shape in the collision, and direction, where the shape moves after the hit; leave direction out if it stops. direction is one of {_DIRECTIONS}. East is screen-right and south is screen-down."""

_EXAMPLE = """Example response:

[
  {"time": 0.0, "events": [
    {"shape": "square", "color": "red", "doing": "still"},
    {"shape": "circle", "color": "blue", "doing": "moving", "direction": "east"}
  ]},
  {"time": 0.7, "events": [
    {"shape": "square", "color": "red", "doing": "colliding", "direction": "east", "with": {"shape": "circle", "color": "blue"}},
    {"shape": "circle", "color": "blue", "doing": "colliding", "with": {"shape": "square", "color": "red"}}
  ]},
  {"time": 1.4, "events": [
    {"shape": "square", "color": "red", "doing": "exiting", "direction": "east"}
  ]}
]"""

_DEFINITIONS = """Definitions:
- entering: a shape that was outside the frame comes fully into view through an edge.
- exiting: a shape that was in view starts leaving through an edge.
- colliding: two shapes touch. Both shapes get an entry at the same time, each naming the other in with.
- A still shape never moves unless something hits it. A moving shape keeps its direction until it hits something."""


def base() -> str:
    return (
        "Watch the video of colored shapes on a white background. Describe only shapes that are "
        "clearly on screen.\n\n" + _FORMAT + "\n\nDo not mention a shape that is off screen, and "
        "do not guess where it went.\n\n" + _EXAMPLE
    )


def defined() -> str:
    return (
        "Watch the video of colored shapes on a white background. Describe only shapes that are "
        "clearly on screen.\n\n" + _FORMAT + "\n\n" + _DEFINITIONS + "\n\nDo not mention a shape "
        "that is off screen, and do not guess where it went.\n\n" + _EXAMPLE
    )


def inventory() -> str:
    """Ask for a count of shapes first, inside the JSON, to anchor the listing."""
    return defined().replace(
        "Reply with a JSON list and nothing else.",
        "First look at the whole video. Reply with a JSON list and nothing else.",
    ) + "\n\nBefore answering, check every frame for collisions: shapes that touch even once collide."


_CLOSED = f"""Reply with a JSON list and nothing else. Each item has a time, in seconds, and an events array.

The first item has time 0.0 and lists every shape that is on screen at the start, each as still or moving. Describe the start only: a shape that is not moving at the start is still, even if something hits it later.
After that, add an item only when a shape is entering, exiting, or colliding, and list only the shapes doing that. Do not add items where shapes just keep moving.

A shape entry has shape, color, and doing.
- shape is exactly one of: {_SHAPES}. There are no other shapes. A square turned at an angle is still a square, never a diamond.
- color is exactly one of: {_COLORS}.
- doing is one of: still, moving, entering, exiting, colliding.
moving, entering, and exiting include direction. colliding includes with, the other shape in the collision, and direction, where the shape moves after the hit; leave direction out if it stops. A shape that only drifts very slowly counts as still.
direction is exactly one of: {_DIRECTIONS}. East is screen-right and south is screen-down. Pick the closest."""


def closed() -> str:
    """Closed lists of shapes, colors, and directions. Still is judged on the first frame."""
    return (
        "Watch the video of colored shapes on a white background. Describe only shapes that are "
        "clearly on screen.\n\n" + _CLOSED + "\n\n" + _DEFINITIONS + "\n\nDo not mention a shape "
        "that is off screen, and do not guess where it went.\n\n" + _EXAMPLE
    )


def closed_inventory() -> str:
    """closed, plus a pass over the whole clip before answering."""
    return closed().replace(
        "Reply with a JSON list and nothing else.",
        "First look at the whole video. Reply with a JSON list and nothing else.",
    ) + "\n\nBefore answering, check every frame for collisions: shapes that touch even once collide."


def half_second() -> str:
    """closed_inventory, but still versus moving is judged over the first half second."""
    return closed_inventory().replace(
        "Describe the start only: a shape that is not moving at the start is still, even if something hits it later.",
        "Judge each shape by how it moves during the first half second. A shape that moves then is moving, "
        "including a shape that is heading toward another shape. A shape that does not move then is still, "
        "even if something hits it later.",
    ).replace(" A shape that only drifts very slowly counts as still.", "")


PROMPTS = {
    "base": base,
    "defined": defined,
    "inventory": inventory,
    "closed": closed,
    "closed_inventory": closed_inventory,
    "half_second": half_second,
}
