"""Логіка кола: обов'язковий порядок checkpoints, таймер, запис і відтворення
найкращого заїзду (ghost).

Checkpoint зараховується тільки коли машина знаходиться близько до нього і
це наступний за порядком (не можна проскочити чи зарахувати не по черзі).
Коло завершується, коли пройдені всі checkpoints і машина повернулась до
checkpoint 0 (він же старт/фініш).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHECKPOINT_RADIUS = 30.0  # світові одиниці — наскільки близько треба підʼїхати
START_MOVE_SPEED = 2.0    # одиниць/сек — поріг швидкості, що вважається "рушив з місця"

GAMEPLAY_LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "gameplay"


@dataclass
class GhostFrame:
    position: tuple[float, float]
    heading: float


@dataclass
class LapTracker:
    checkpoints: np.ndarray  # (M, 2) світові координати checkpoints по порядку
    next_checkpoint_idx: int = 0
    lap_time: float = 0.0
    lap_running: bool = False
    best_time: float | None = None
    best_ghost: list[GhostFrame] = field(default_factory=list)
    _current_recording: list[GhostFrame] = field(default_factory=list)

    def reset_progress(self) -> None:
        """Скидає прогрес поточного кола (напр. при спавні машини), не чіпаючи
        збережений найкращий заїзд/час."""
        self.next_checkpoint_idx = 0
        self.lap_time = 0.0
        self.lap_running = False
        self._current_recording = []

    def update(self, position: np.ndarray, heading: float, dt: float, speed: float = 0.0) -> bool:
        """Викликається щокадру. Повертає True, якщо саме в цьому кадрі
        завершилось повне коло (усі checkpoints по порядку + повернення на 0)."""
        n = len(self.checkpoints)

        if not self.lap_running:
            # Коло (і запис ghost) стартує в момент, коли гравець РЕАЛЬНО почав
            # рухатись, а не коли машина щойно з'явилась на checkpoint 0 — інакше
            # час "роздумів перед стартом" потрапляв у lap_time, а ghost показував
            # машину, що спочатку стоїть на місці.
            if speed >= START_MOVE_SPEED:
                self.lap_running = True
                self.lap_time = 0.0
                self._current_recording = []
                self.next_checkpoint_idx = 1 % n
            return False

        self.lap_time += dt
        self._current_recording.append(
            GhostFrame(position=(float(position[0]), float(position[1])), heading=heading)
        )

        target = self.checkpoints[self.next_checkpoint_idx]
        dist = float(np.linalg.norm(position - target))
        lap_completed = False

        if dist <= CHECKPOINT_RADIUS:
            if self.next_checkpoint_idx == 0:
                # Повний оберт: усі проміжні checkpoints вже пройдені (next дійшов
                # циклічно назад до 0), і машина фізично на checkpoint 0 — фініш.
                lap_completed = True
                self._finish_lap()
            else:
                self.next_checkpoint_idx = (self.next_checkpoint_idx + 1) % n

        return lap_completed

    def _finish_lap(self) -> None:
        finished_time = self.lap_time
        finished_recording = self._current_recording

        if self.best_time is None or finished_time < self.best_time:
            self.best_time = finished_time
            self.best_ghost = finished_recording

        # Одразу починаємо нове коло (гравець "нескінченно" кружляє, як і просив).
        self.lap_time = 0.0
        self._current_recording = []
        self.next_checkpoint_idx = 1 % len(self.checkpoints)

    def save_best_to_disk(self, seed: int) -> None:
        """Зберігає найкращий заїзд у logs/gameplay/seed_<seed>/best_lap.json —
        при повторному запуску гри з тим самим seed результат не втрачається."""
        if self.best_time is None:
            return
        seed_dir = GAMEPLAY_LOG_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "best_time": self.best_time,
            "ghost": [
                {"position": list(f.position), "heading": f.heading}
                for f in self.best_ghost
            ],
        }
        with open(seed_dir / "best_lap.json", "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def load_best_from_disk(self, seed: int) -> bool:
        """Завантажує збережений найкращий заїзд для цього seed, якщо є.
        Повертає True, якщо щось реально завантажено."""
        path = GAMEPLAY_LOG_DIR / f"seed_{seed}" / "best_lap.json"
        if not path.exists():
            return False
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.best_time = float(data["best_time"])
            self.best_ghost = [
                GhostFrame(position=tuple(f["position"]), heading=float(f["heading"]))
                for f in data["ghost"]
            ]
            return True
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            # Пошкоджений чи несумісний файл — не валимо гру, просто ігноруємо.
            return False


@dataclass
class GhostPlayer:
    """Відтворює записаний найкращий заїзд синхронно з поточним часом кола
    гравця — не власним таймером, а "прив'язаний" до lap_time трекера, щоб
    привид завжди показував "де я був у той самий момент часу минулого разу"."""
    frames: list[GhostFrame]

    def sample(self, lap_time: float, fps_hint: float = 60.0) -> GhostFrame | None:
        if not self.frames:
            return None
        idx = int(lap_time * fps_hint)
        if idx >= len(self.frames):
            return None  # привид уже доїхав до фінішу — зникає, чекає нового кола
        return self.frames[idx]
