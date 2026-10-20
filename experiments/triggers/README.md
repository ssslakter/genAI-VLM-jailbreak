# Triggers

How many generated clips realize each rung of a trigger ladder, and how many are near-misses?

This does not call a model. `count.py` reads the ground-truth logs and prints, per dataset directory, the share of clips in each bin:

- presence of a circle, the color red, a collision
- conjunction: a red circle, versus red and circle split across shapes
- set: a collision and an exit, versus only one of them
- sequence: a collision and then an exit, versus the reverse order or the two within 0.1 s
- set and sequence on one pair of events, a shape leaving through the top edge and one leaving through the left edge: both present, versus one of them; north before west, versus the reverse order

Events of different kinds keep one order in these 2 s clips (an entry comes before the collision it
causes, the collision before the exit it causes): across 10 000 clips an exit is followed by an
entry of another shape in 35 and by a collision in 158. Two exits in different directions come in
either order (north then west 247, west then north 224), so the ordered rung uses those.

```bash
pixi run python experiments/triggers/count.py dataset
```

## Results

None checked in yet.
