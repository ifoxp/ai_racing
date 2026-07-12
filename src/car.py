"""Фізика машини — напівреалістична аркадна модель (bicycle model + traction).

Ключова ідея: вектор швидкості (velocity) і напрямок капота (heading) —
РІЗНІ речі. Газ штовхає машину вздовж heading, але фактичний рух іде вздовж
velocity. "Зчеплення" (traction) поступово підтягує velocity до heading —
на низькій швидкості це відбувається майже миттєво (машина їде рівно туди,
куди повернуто колеса), а на високій швидкості повільніше — звідси занос
на швидких поворотах, а не миттєвий розворот.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- Налаштування фізики (підібрані під світові одиниці треку: ширина ~44) ---
MAX_SPEED = 220.0            # одиниць/сек
ACCELERATION = 140.0         # одиниць/сек^2, поки газ=1 (розгін 0->max ~1.6с, було ~0.85с)
BRAKE_DECELERATION = 340.0   # одиниць/сек^2 * (сила гальма 0..1)
DRAG = 40.0                  # природне сповільнення без газу, одиниць/сек^2
MAX_STEER_RATE = 3.6         # рад/сек — максимальна кутова швидкість повороту капота на швидкості=0
MIN_STEER_RATE = 1.1         # рад/сек — на MAX_SPEED (вдвічі чутливіше, ніж було, — менш "туго")
BASE_TRACTION = 10.0         # 1/сек — на швидкості=0: velocity миттєво = heading
MIN_TRACTION = 1.6           # 1/сек — на MAX_SPEED: явний занос, повільне підтягування
CAR_LENGTH = 14.0
CAR_WIDTH = 7.0


@dataclass
class CarState:
    position: np.ndarray  # (2,) світові координати центру машини
    heading: float         # радіани, напрямок капота (0 = вздовж +X)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (2,) вектор фактичного руху

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def corners(self) -> np.ndarray:
        """4 кути хітбоксу машини у світових координатах (для колізій з межею
        траси на майбутньому етапі — reward-функція з track limits)."""
        hl, hw = CAR_LENGTH / 2, CAR_WIDTH / 2
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        c, s = np.cos(self.heading), np.sin(self.heading)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + self.position


def step_car(state: CarState, throttle: float, steer: float, brake: float, dt: float) -> CarState:
    """Один крок фізики. throttle: 0 або 1 (бінарний газ). steer: -1..1
    (вліво/вправо). brake: 0..1 (сила гальмування). dt: секунди."""
    speed = state.speed
    speed_frac = min(speed / MAX_SPEED, 1.0)

    # --- Кермо: чим вища швидкість, тим повільніше повертається капот ---
    steer_rate = MAX_STEER_RATE + (MIN_STEER_RATE - MAX_STEER_RATE) * speed_frac
    new_heading = state.heading + steer * steer_rate * dt

    heading_vec = np.array([np.cos(new_heading), np.sin(new_heading)])

    # --- Поздовжня сила: газ вздовж heading, гальмо проти velocity, тертя завжди проти velocity ---
    velocity = state.velocity.copy()
    if throttle > 0:
        velocity += heading_vec * ACCELERATION * dt

    if speed > 1e-6:
        vel_dir = velocity / max(np.linalg.norm(velocity), 1e-6)
        # Гальмо: сповільнює вздовж поточного напрямку руху
        brake_decel = BRAKE_DECELERATION * brake * dt
        drag_decel = DRAG * dt
        total_decel = brake_decel + drag_decel
        new_speed = max(0.0, np.linalg.norm(velocity) - total_decel)
        velocity = vel_dir * new_speed

    # --- Зчеплення: підтягує напрямок velocity до heading. Чим вища швидкість,
    # тим слабше підтягування — на низькій швидкості машина "чіпляється" за
    # дорогу і їде рівно туди, куди повернуті колеса; на високій — заносить. ---
    cur_speed = np.linalg.norm(velocity)
    if cur_speed > 1e-6:
        traction = BASE_TRACTION + (MIN_TRACTION - BASE_TRACTION) * speed_frac
        vel_dir = velocity / cur_speed
        blend = min(traction * dt, 1.0)
        new_dir = vel_dir * (1 - blend) + heading_vec * blend
        norm = np.linalg.norm(new_dir)
        if norm > 1e-6:
            velocity = (new_dir / norm) * cur_speed

    # Обмеження максимальної швидкості (після додавання газу вона могла перевищити)
    cur_speed = np.linalg.norm(velocity)
    if cur_speed > MAX_SPEED:
        velocity = velocity / cur_speed * MAX_SPEED

    new_position = state.position + velocity * dt

    return CarState(position=new_position, heading=new_heading, velocity=velocity)
