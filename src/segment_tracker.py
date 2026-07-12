"""Спільна для всіх ботів статистика "які сегменти траси стабільно вбивають".

Сегмент — ділянка асфальту між двома сусідніми checkpoints (індексується за
checkpoint, до якого машина їхала). Кожен бот у своєму окремому процесі
(SubprocVecEnv) звітує "помер ТУТ" чи "пройшов САМЕ ЦЕЙ сегмент" через info у
CarRacingEnv.step() — оцінка йде per-сегмент, не за ціле коло: на трасі з
кількома складними поворотами шанс пройти ВСЮ дистанцію без жодної помилки
одразу мізерний, і "проблемна позначка" на одному конкретному повороті мала б
тоді ніколи не зникати, навіть якщо саме цей поворот уже стабільно
проїжджається. BackgroundTrainer (батьківський процес) збирає ці звіти сюди,
а SegmentDifficultyTracker вираховує ковзну частку смертей за останні
WINDOW_SIZE спроб на кожен сегмент окремо. Коли частка перевищує поріг —
сегмент вважається "проблемним" і отримує додатковий штраф; коли бот навчився
стабільно проходити САМЕ цей сегмент — штраф на ньому сам згасає.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

WINDOW_SIZE = 20          # скільки останніх спроб на сегмент враховується
DEATH_RATE_THRESHOLD = 0.3  # від цієї частки смертей у вікні сегмент вважається проблемним
MIN_SAMPLES = 5            # не робити висновків, поки на сегмент менше спроб
EXTRA_PENALTY = -3.0       # додатковий штраф за смерть у проблемному сегменті


@dataclass
class SegmentDifficultyTracker:
    _history: dict[int, deque[bool]] = field(default_factory=dict)  # segment_idx -> [True=death, False=pass]

    def record_death(self, segment_idx: int) -> None:
        self._history.setdefault(segment_idx, deque(maxlen=WINDOW_SIZE)).append(True)

    def record_pass(self, segment_idx: int) -> None:
        """Саме цей сегмент щойно пройдено без вильоту — незалежно від того,
        що відбувається на решті траси."""
        self._history.setdefault(segment_idx, deque(maxlen=WINDOW_SIZE)).append(False)

    def death_rate(self, segment_idx: int) -> float:
        history = self._history.get(segment_idx)
        if not history or len(history) < MIN_SAMPLES:
            return 0.0
        return sum(history) / len(history)

    def is_problematic(self, segment_idx: int) -> bool:
        return self.death_rate(segment_idx) >= DEATH_RATE_THRESHOLD

    def problematic_segments(self) -> dict[int, float]:
        """segment_idx -> death_rate, лише для сегментів понад поріг."""
        return {idx: rate for idx in self._history if (rate := self.death_rate(idx)) >= DEATH_RATE_THRESHOLD}

    def penalties(self) -> dict[int, float]:
        """segment_idx -> extra_penalty, готове для CarRacingEnv.set_segment_penalties()."""
        return {idx: EXTRA_PENALTY for idx in self.problematic_segments()}
