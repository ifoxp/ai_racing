"""Режим "Заїзди" — паралельний прогін кількох УЖЕ НАВЧЕНИХ моделей на одній
трасі, без навчання. На відміну від BackgroundTrainer (SubprocVecEnv, фоновий
потік, PPO.learn()), тут кожна модель — окремий CarRacingEnv, що крутиться
напряму в головному потоці Arcade (той самий патерн, яким зараз рухається
машина гравця через step_car() у main.py.on_update()). Inference (model.predict)
— мілісекунди, тому 2-8 моделей одночасно в одному процесі без проблем з
продуктивністю, жодних OS-підпроцесів не потрібно.

Виліт за межі не "вбиває" машину назавжди — просто миттєвий respawn на
старті (як і планувалось раніше для навчання при зміні траси: "ніби зникла
остання ітерація"), інакше глядач лишився б з порожньою трасою вже за
кілька секунд перегляду."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import arcade
from stable_baselines3 import PPO

import theme
from rl_env import CarRacingEnv
from track_generator import Track
from trainer import RenderSnapshot

ROW_HEIGHT = 30
MAX_LIST_HEIGHT = 260

# Палітра відрізняється від theme.CYAN/ACCENT (ті зайняті ботами навчання й
# гравцем) — фіксований набір насичених, легко розрізнюваних кольорів, а не
# випадковий RGB (той міг би вийти нечитабельним на асфальті чи майже
# однаковим для двох машин).
RACE_CAR_COLORS: list[tuple[int, int, int]] = [
    (232, 90, 90),    # червоний
    (95, 191, 143),   # зелений
    (94, 150, 232),   # синій
    (214, 149, 232),  # фіолетовий
    (232, 200, 90),   # жовтий
    (90, 214, 214),   # бірюзовий
    (232, 140, 90),   # оранжевий
    (180, 200, 100),  # оливковий
]


@dataclass
class RaceCarSlot:
    """Один учасник заїзду: завантажена модель + власне середовище фізики.
    reward з env.step() ігнорується повністю — тут не навчання, лише
    прогін готової політики. Поточний час кола і номер кола не дублюються
    тут — вони й так живуть у env (episode_time/laps_completed), лишень
    найкращий час бота рахуємо окремо, бо CarRacingEnv його не зберігає."""
    name: str
    color: tuple[int, int, int]
    model: PPO
    env: CarRacingEnv
    best_lap_time: float | None = None


class RaceSession:
    """Тримає список RaceCarSlot і крутить їх усіх на один крок фізики за
    викликом step() — так само як BackgroundTrainer тримає VecEnv, але без
    потоків/підпроцесів: усе відбувається синхронно в тому самому кадрі
    Arcade, бо inference на 2-8 моделей займає частки мілісекунди."""

    def __init__(self, track: Track, model_paths: list[Path]):
        self.track = track
        self.slots: list[RaceCarSlot] = []

        # Колір кожної машини — випадковий з фіксованої палітри, без
        # повторів, поки моделей не більше, ніж кольорів у палітрі (типовий
        # заїзд 2-8 моделей, кольорів 8 — повтір трапиться лише на 9+ учасників).
        colors = RACE_CAR_COLORS.copy()
        random.shuffle(colors)
        for i, path in enumerate(model_paths):
            env = CarRacingEnv(track)
            env.reset()
            model = PPO.load(str(path), device="cpu")
            color = colors[i % len(colors)]
            self.slots.append(RaceCarSlot(name=path.stem, color=color, model=model, env=env))

    def set_track(self, track: Track) -> None:
        """Нова траса — переспавнює всіх учасників на нових координатах,
        той самий патерн, що CarRacingEnv.set_track() уже робить для одного
        env всередині BackgroundTrainer."""
        self.track = track
        for slot in self.slots:
            slot.env.set_track(track)
            slot.best_lap_time = None

    def step(self, dt: float) -> None:
        for slot in self.slots:
            obs = slot.env._get_obs()
            action, _ = slot.model.predict(obs, deterministic=True)
            _, _, terminated, truncated, info = slot.env.step(action)

            lap_time = info.get("lap_time")
            if lap_time is not None and (slot.best_lap_time is None or lap_time < slot.best_lap_time):
                slot.best_lap_time = lap_time

            if terminated or truncated:
                # Виліт або кінець епізоду (MAX_LAPS_PER_EPISODE) — тихий
                # respawn, глядач бачить лише миттєве "перестрибування" машини
                # на старт, без жодного штрафу чи паузи (тут немає reward).
                slot.env.reset()

    def get_snapshots(self) -> list[RenderSnapshot]:
        snapshots = []
        for i, slot in enumerate(self.slots):
            car = slot.env.car
            if car is None:
                continue
            snapshots.append(RenderSnapshot(
                bot_id=i,
                name=slot.name,
                position=car.position,
                heading=car.heading,
                ray_distances=slot.env._last_rays,
                current_lap_time=slot.env.episode_time,
                lap_number=slot.env.laps_completed + 1,
                best_lap_time=slot.best_lap_time,
            ))
        return snapshots


def list_available_models(models_dir: Path) -> list[Path]:
    """Заїзди мають бачити і ручні збереження (logs/models/*.zip), і
    автозбереження (logs/models/saves/*.zip) — на відміну від
    _list_saved_models() у main.py (лише для "Завантажити" в навчанні),
    де autosave свідомо не показуються (там завжди явно назване збереження)."""
    manual = list(models_dir.glob("*.zip"))
    autosaves = list((models_dir / "saves").glob("*.zip"))
    return sorted(manual + autosaves, key=lambda p: p.name.lower())


class ModelPickerPanel:
    """Список файлів моделей із чекбоксами для режиму "Заїзди" — той самий
    патерн, що BotPanel (заголовок, скрол, пул arcade.Text, _row_bounds для
    кліків), але мультивибір замість виділення одного бота, і без сортування
    за reward (моделі просто в алфавітному порядку файлів)."""

    def __init__(self, x: float, y: float, width: float = 260):
        self.x = x
        self.y = y
        self.width = width
        self.scroll_offset = 0.0
        self.selected: set[str] = set()  # шляхи (як str) обраних моделей

        self._title_text = arcade.Text(
            "МОДЕЛІ ДЛЯ ЗАЇЗДУ", x + 12, y - 18, theme.ACCENT, font_size=11,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )
        self._empty_text = arcade.Text(
            "немає збережених моделей у logs/models", x + 12, y - 46, theme.TEXT_FAINT,
            font_size=11, anchor_x="left", anchor_y="top", font_name=theme.FONT,
        )
        self._row_checkbox_texts: list[arcade.Text] = []  # пул "✓"/"" на рядок
        self._row_name_texts: list[arcade.Text] = []
        self._row_bounds: list[tuple[str, float, float, float, float]] = []  # (path_str, left, right, bottom, top)

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._title_text.x = x + 12
        self._title_text.y = y - 18
        self._empty_text.x = x + 12
        self._empty_text.y = y - 46

    def _header_bounds(self) -> tuple[float, float, float, float]:
        return self.x, self.x + self.width, self.y - 30, self.y

    def _ensure_row_pool(self, n: int) -> None:
        while len(self._row_checkbox_texts) < n:
            self._row_checkbox_texts.append(arcade.Text(
                "", 0, 0, theme.ACCENT_STRONG, font_size=13,
                anchor_x="center", anchor_y="center", font_name=theme.FONT, bold=True,
            ))
            self._row_name_texts.append(arcade.Text(
                "", 0, 0, theme.TEXT_PRIMARY, font_size=12,
                anchor_x="left", anchor_y="center", font_name=theme.FONT,
            ))

    def draw(self, models: list[Path]) -> None:
        header_left, header_right, header_bottom, header_top = self._header_bounds()
        arcade.draw_lrbt_rectangle_filled(header_left, header_right, header_bottom, header_top, theme.BG_RAISED + (215,))
        arcade.draw_lrbt_rectangle_outline(header_left, header_right, header_bottom, header_top, theme.BORDER_SOFT, border_width=1.5)
        self._title_text.draw()

        self._row_bounds = []

        if not models:
            list_bottom = header_bottom - 34
            arcade.draw_lrbt_rectangle_filled(header_left, header_right, list_bottom, header_bottom, theme.BG_RAISED + (215,))
            arcade.draw_lrbt_rectangle_outline(header_left, header_right, list_bottom, header_bottom, theme.BORDER_SOFT, border_width=1.5)
            self._empty_text.draw()
            return

        list_height = min(len(models) * ROW_HEIGHT, MAX_LIST_HEIGHT)
        max_scroll = max(0.0, len(models) * ROW_HEIGHT - MAX_LIST_HEIGHT)
        self.scroll_offset = max(0.0, min(self.scroll_offset, max_scroll))

        list_top = header_bottom
        list_bottom = list_top - list_height
        arcade.draw_lrbt_rectangle_filled(header_left, header_right, list_bottom, list_top, theme.BG_RAISED + (215,))
        arcade.draw_lrbt_rectangle_outline(header_left, header_right, list_bottom, list_top, theme.BORDER_SOFT, border_width=1.5)

        self._ensure_row_pool(len(models))

        for i, path in enumerate(models):
            row_y = list_top - i * ROW_HEIGHT + self.scroll_offset - ROW_HEIGHT / 2
            if row_y > list_top or row_y < list_bottom - ROW_HEIGHT:
                continue

            row_top = min(row_y + ROW_HEIGHT / 2, list_top)
            row_bottom = max(row_y - ROW_HEIGHT / 2, list_bottom)
            row_valid = row_bottom < row_top
            path_str = str(path)
            if row_valid:
                self._row_bounds.append((path_str, header_left, header_right, row_bottom, row_top))

            is_checked = path_str in self.selected
            if is_checked and row_valid:
                arcade.draw_lrbt_rectangle_filled(header_left, header_right, row_bottom, row_top, theme.ACCENT_SOFT)

            checkbox_text, name_text = self._row_checkbox_texts[i], self._row_name_texts[i]
            box_cx = header_left + 20
            box_h = 8.0
            arcade.draw_lrbt_rectangle_outline(
                box_cx - box_h, box_cx + box_h, row_y - box_h, row_y + box_h,
                theme.ACCENT_STRONG if is_checked else theme.BORDER, border_width=1.5,
            )
            checkbox_text.text = "✓" if is_checked else ""
            checkbox_text.x = box_cx
            checkbox_text.y = row_y

            name_text.text = path.stem
            name_text.x = header_left + 38
            name_text.y = row_y

            if list_bottom <= row_y <= list_top:
                checkbox_text.draw()
                name_text.draw()

    def on_mouse_press(self, x: float, y: float) -> None:
        for path_str, left, right, bottom, top in self._row_bounds:
            if left <= x <= right and bottom <= y <= top:
                if path_str in self.selected:
                    self.selected.discard(path_str)
                else:
                    self.selected.add(path_str)
                return

    def on_mouse_scroll(self, x: float, y: float, scroll_y: int) -> None:
        header_left, header_right, header_bottom, _ = self._header_bounds()
        if not (header_left <= x <= header_right):
            return
        self.scroll_offset -= scroll_y * ROW_HEIGHT * 1.5

    def selected_paths(self) -> list[Path]:
        return [Path(p) for p in sorted(self.selected)]
