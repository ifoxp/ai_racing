**English** · [Українська](README.uk.md)

# ai_racing

A procedurally generated race track and a PPO agent that learns to drive it —
training live, on screen, while the game keeps rendering at frame rate.

Built in Python with Arcade, Gymnasium and Stable-Baselines3. No pre-recorded
demonstrations: the agent is not imitating a human lap, it discovers how to
drive from reward alone.

## What it does

- Generates a unique closed circuit from a seed (reproducible tracks)
- Lets you drive it yourself and records the lap as a reference time
- Trains a PPO agent on the same track, with several cars visible at once
- Exposes the training controls — speed, reward weights, sensor angles — in
  the game's own UI, so you can change them and watch what happens

## How the track is generated

`track_generator.py` chains five steps, each chosen to fix a specific problem
with the previous one:

1. **Voronoi** — scatter seed points, get cells
2. **Delaunay** — turn cells into a graph of plausible connections
3. **MST** — reduce it to a spanning tree, which guarantees connectivity
4. **2-opt** — untangle the resulting loop so it stops crossing itself
5. **Catmull-Rom** — smooth the polyline into a drivable curve

The seed makes any track reproducible, which matters because comparing two
training runs on two different tracks tells you nothing.

## How the agent sees the world

The observation is a compact vector, not pixels: raycast distances to the
track edge, current speed, plus direction and distance to the next checkpoint.
Compact input learns fast and needs no GPU for the observation itself.

The default ray fan is deliberate, and every angle has a reason:

| Angles | Why |
|---|---|
| ±5°, ±20°, ±40°, ±65° | forward, with the step widening gradually so there is no blind gap where a corner entry appears |
| ±90° | the sides |
| ±110°, ±140° | rear diagonals — during a drift the tail slides out into exactly this band |

There is **no ray at 0°**, because it would duplicate ±5°, and **none at
180°**, because it stares at road already travelled and reads as a near
constant.

The ray *count* is frozen. It defines the observation size, so changing it
invalidates every saved model. The angles themselves are editable from the
settings tab.

## Reward shaping, and what it took to get there

The reward function is the part that went through the most iteration. Three
decisions worth explaining:

**The action space was cut from 132 to 44.** Originally throttle × 11 steering
levels × 6 brake levels. Braking became binary, matching throttle. With 132
combinations the agent almost never stumbled onto a working action for a 90°
corner within a five-minute session; with 44 it does.

**The brake bonus scales with current speed** rather than being a flat value.
At low speed there is no reason to brake — the dense reward from slowing down
already outweighs any fixed bonus. At high speed, braking is exactly what
decides whether the car makes the corner. Without this, PPO reliably found the
lazy local optimum: lift off the throttle and let drag do the work, never
discovering throttle + brake + steer together.

**Completing a lap no longer ends the episode.** The agent keeps driving and
keeps its accumulated reward, which turns the incentive into "protect and grow
your score" instead of "finish and reset". Episodes end on leaving the track,
after three clean laps, or on timeout — the boundary has to exist somewhere,
or PPO's rollout buffer never sees an episode end and the algorithm loses the
notion of where the task stops.

## The threading problem, and why it is solved this way

This is the part that is easy to get wrong.

`model.learn()` blocks. Called on the main thread, it freezes the Arcade
window for the entire training run — no rendering, no input, an application
that looks crashed.

So training runs on a **background thread**, and the two threads exchange
state through a mutex-guarded list of render snapshots: the training thread
writes every car's state after each environment step, the main thread reads it
once per frame to draw.

N parallel copies of the track run in `SubprocVecEnv`, each in its own OS
process, feeding one shared rollout buffer for a **single** policy. That is
why several cars appear at once — they are not competing, they are parallel
experience collectors for one brain.

`SubprocVecEnv` is created **once, lazily, inside the background thread** on
the first `start()` — never in the constructor, never on the main thread.
Spawning N processes on Windows is slow: there is no `fork`, so each process
is a cold interpreter start. Doing it synchronously on the main thread froze
the window visibly, for seconds, every time a new track was generated.

When the track changes and the workers are already running, the new geometry
is pushed into them with `env_method("set_track", ...)` instead of tearing
everything down and paying the spawn cost again.

## Running it

```bash
pip install -r requirements.txt
python src/main.py
```

PyTorch with CUDA is optional — training works on CPU, just slower.

## Project state

Five of seven planned stages are done: track generation and UI, manual
driving, lap recording, RL with live visualisation, and training controls.
Multi-agent racing and player-versus-bots are not built yet.

The `Навчання/` directory holds hand-written HTML pages explaining the Arcade
library, the track generation chain and the RL algorithms — written while
learning them, kept because they are useful.

---

**Chekaliuk Dmytro** · [@ifoxp](https://github.com/ifoxp) ·
[ifoxp.top](https://ifoxp.top) · Telegram [@ifoxp](https://t.me/ifoxp) ·
Discord `ifoxp`
