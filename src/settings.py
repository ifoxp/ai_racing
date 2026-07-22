"""Налаштування нагород і сенсорів — єдине джерело правди для всіх тюнінг-
параметрів навчання, редагованих у вкладці "Налаштування".

Чому окремий модуль, а не константи в rl_env.py: константи "запечені" в код,
а користувач хоче експериментувати з цифрами прямо з гри (поля вводу + кнопка
скидання до дефолту біля кожного). Дефолти лишаються констанами тут-таки —
кнопка "скинути" повертає саме до них.

Зберігається в logs/config.json — переживає перезапуск гри.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent.parent / "logs" / "config.json"

# Половина віяла променів (тільки додатні кути) — повний набір це ±кожен з
# них, симетрично. Користувач може міняти градуси, але НЕ кількість:
# кількість променів визначає розмір observation-вектора, і будь-яка зміна
# кількості робить усі збережені моделі несумісними. Зміна лише кутів
# зберігає розмірність — стара модель працюватиме (хоч і бачитиме світ
# трохи інакше, ніж під час навчання).
DEFAULT_RAY_HALF_ANGLES: tuple[float, ...] = (5.0, 20.0, 40.0, 65.0, 90.0, 110.0, 140.0)


@dataclass
class RewardConfig:
    """Абсолютні значення компонентів reward (не множники — після переїзду
    налаштувань у власну вкладку проміжний шар "ваг" став зайвим: поле
    показує саме те число, що піде в формулу)."""
    checkpoint_bonus: float = 1.0        # одноразово за пройдений checkpoint
    lap_bonus_scale: float = 10.0        # множник бонусу за коло: min(1, реф/час) * це
    speed_reward_scale: float = 0.002    # щокадрово: це * поточна швидкість
    brake_bonus_scale: float = 0.01      # щокадрово в зоні гострого повороту: це * швидкість
    stuck_penalty: float = -0.05         # щокадрово після 2с "застрягання"
    out_of_bounds_penalty: float = -10.0  # при вильоті (разово якщо смерть, щокроку якщо ні)
    out_of_bounds_is_death: bool = True  # False = штраф щокроку за межею, але епізод живе
    score_death_threshold: float = -1000.0  # сумарний reward епізоду нижче цього = смерть
    ray_half_angles: list[float] = field(default_factory=lambda: list(DEFAULT_RAY_HALF_ANGLES))

    def full_ray_angles(self) -> np.ndarray:
        """Повний набір кутів (±кожен). ПОРЯДОК КРИТИЧНИЙ і не має права
        змінюватись: це порядок компонент observation-вектора, під який
        навчені ваги мережі. Історичний порядок (з часів, коли кути були
        константою в rl_env.py): спершу переднє віяло (кути ≤90°, від'ємні
        за спаданням модуля, потім додатні за зростанням), потім задні
        (>90°, так само). Одного разу цей порядок уже випадково змінився
        на "просто відсортовані" — і всі навчені моделі різко подурнішали,
        бо їхні входи переставились місцями (знайшов користувач)."""
        front = [a for a in self.ray_half_angles if a <= 90.0]
        rear = [a for a in self.ray_half_angles if a > 90.0]
        ordered = ([-a for a in reversed(front)] + front
                   + [-a for a in reversed(rear)] + rear)
        return np.array(ordered, dtype=np.float64)


def load_config() -> RewardConfig:
    """Читає config.json; відсутні ключі (стара версія файлу після додавання
    нових полів) заповнюються дефолтами, зайві — ігноруються."""
    if not CONFIG_PATH.exists():
        return RewardConfig()
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return RewardConfig()
    known = {f.name for f in fields(RewardConfig)}
    clean = {k: v for k, v in raw.items() if k in known}
    cfg = RewardConfig(**clean)
    # Кількість променів фіксована — пошкоджений/старий список кутів у файлі
    # не має права зламати розмір observation.
    if len(cfg.ray_half_angles) != len(DEFAULT_RAY_HALF_ANGLES):
        cfg.ray_half_angles = list(DEFAULT_RAY_HALF_ANGLES)
    return cfg


def save_config(cfg: RewardConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
