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

import base64
import json
import pickle
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import arcade
from stable_baselines3 import PPO

import theme
from rl_env import CarRacingEnv
from settings import RewardConfig
from track_generator import Track
from trainer import RenderSnapshot
from ui_panel import Button

ROW_HEIGHT = 38  # два піврядки: ім'я + лічильник ×N зверху, кроки навчання знизу
MAX_LIST_HEIGHT = 300
MAX_INSTANCES_PER_MODEL = 8

# Кеш метаданих моделей: читання zip щокадру в draw() було б марнотратством.
# Ключ — шлях, значення — (mtime, кроки, obs_dim); mtime інвалідовує кеш,
# якщо файл перезаписано новішою моделлю з тією ж назвою.
_meta_cache: dict[str, tuple[float, int | None, int | None]] = {}


def read_model_meta(path: Path) -> tuple[int | None, int | None]:
    """(кроки навчання, розмір observation) зі збереженого PPO.zip БЕЗ
    повного PPO.load (той тягне torch-ваги — повільно для списку файлів).
    SB3-архів містить 'data' (JSON): num_timesteps лежить звичайним числом,
    observation_space — base64-піклом. Якщо щось не читається — (None, None):
    краще показати модель без метаданих, ніж впасти на списку файлів."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, None
    cached = _meta_cache.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]

    steps: int | None = None
    obs_dim: int | None = None
    try:
        with zipfile.ZipFile(path) as zf:
            data = json.loads(zf.read("data").decode("utf-8"))
        raw_steps = data.get("num_timesteps")
        if isinstance(raw_steps, int):
            steps = raw_steps
        serialized = data.get("observation_space")
        if isinstance(serialized, dict) and ":serialized:" in serialized:
            space = pickle.loads(base64.b64decode(serialized[":serialized:"]))
            obs_dim = int(space.shape[0])
    except Exception:
        pass
    _meta_cache[str(path)] = (mtime, steps, obs_dim)
    return steps, obs_dim


def _format_steps(steps: int | None) -> str:
    if steps is None:
        return "кроки: ?"
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}M кроків"
    if steps >= 1_000:
        return f"{steps / 1_000:.0f}K кроків"
    return f"{steps} кроків"


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

    def __init__(self, track: Track, model_paths: list[Path], config: RewardConfig | None = None):
        """model_paths МОЖЕ містити дублікати — кілька екземплярів однієї
        моделі в одному заїзді (вони не навчаються і не діляться станом,
        лише читають ту саму політику — конфліктів немає). Екземпляри
        отримують суфікси: "Max1", "Max1 #2", "Max1 #3"."""
        self.track = track
        self.slots: list[RaceCarSlot] = []

        # Колір кожної машини — випадковий з палітри, без повторів, поки
        # учасників не більше, ніж кольорів.
        colors = theme.CAR_PALETTE.copy()
        random.shuffle(colors)
        name_counts: dict[str, int] = {}
        # Одна й та сама модель вантажиться з диска ОДИН раз, скільки б
        # екземплярів її не бігало — політика лише читається в predict().
        loaded: dict[str, PPO] = {}
        for i, path in enumerate(model_paths):
            env = CarRacingEnv(track, config=config)
            env.reset()
            key = str(path)
            if key not in loaded:
                loaded[key] = PPO.load(key, device="cpu")
            name_counts[path.stem] = name_counts.get(path.stem, 0) + 1
            n = name_counts[path.stem]
            name = path.stem if n == 1 else f"{path.stem} #{n}"
            color = colors[i % len(colors)]
            self.slots.append(RaceCarSlot(name=name, color=color, model=loaded[key], env=env))

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


# Кеш списку моделей з TTL: list_available_models викликається з on_draw
# ЩОКАДРУ (60/с), а всередині — glob по диску + stat кожного файлу. Понад
# тисяча дискових звернень щосекунди періодично "ікає" на Windows (Defender
# сканує звернення) — це джерело мікрофризів, знайдених користувачем.
# Раз на секунду більш ніж досить: нові збереження з'являються рідко.
_list_cache: dict[tuple[str, int | None], tuple[float, list[Path]]] = {}
_LIST_CACHE_TTL = 1.0


def list_available_models(models_dir: Path, expected_obs_dim: int | None = None) -> list[Path]:
    """Всі збереження: і ручні (logs/models/*.zip), і автозбереження
    (logs/models/saves/*.zip).

    expected_obs_dim: якщо задано — моделі, навчені на ІНШОМУ розмірі
    observation (стара кількість променів), у список не потрапляють: їх
    неможливо запустити, PPO впав би на несумісності розмірів мережі.
    Файли при цьому НЕ видаляються — лише ховаються (на прохання
    користувача: "не видалялась, на всякий випадок"). Моделі з нечитабельними
    метаданими показуються — краще дати спробувати, ніж мовчки сховати."""
    cache_key = (str(models_dir), expected_obs_dim)
    cached = _list_cache.get(cache_key)
    now = time.perf_counter()
    if cached is not None and now - cached[0] < _LIST_CACHE_TTL:
        return cached[1]

    manual = list(models_dir.glob("*.zip"))
    autosaves = list((models_dir / "saves").glob("*.zip"))
    all_models = sorted(manual + autosaves, key=lambda p: p.name.lower())
    if expected_obs_dim is not None:
        result = []
        for path in all_models:
            _, obs_dim = read_model_meta(path)
            if obs_dim is None or obs_dim == expected_obs_dim:
                result.append(path)
        all_models = result
    _list_cache[cache_key] = (now, all_models)
    return all_models


class ModelPickerPanel:
    """Список файлів моделей для режиму "Заїзди" — той самий патерн, що
    BotPanel (заголовок, скрол, пул arcade.Text, _row_bounds для кліків).
    Кожен рядок: чекбокс + назва + лічильник ×N (кнопки −/+ праворуч, можна
    запустити кілька екземплярів однієї моделі) + кроки навчання дрібним
    шрифтом. Несумісні моделі (інший obs_dim) сюди взагалі не потрапляють —
    фільтрує list_available_models за expected_obs_dim."""

    def __init__(self, x: float, y: float, width: float = 260):
        self.x = x
        self.y = y
        self.width = width
        self.scroll_offset = 0.0
        self.counts: dict[str, int] = {}  # шлях (str) -> кількість екземплярів (0 = не обрано)

        self._title_text = arcade.Text(
            "МОДЕЛІ ДЛЯ ЗАЇЗДУ", x + 12, y - 18, theme.ACCENT, font_size=11,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )
        self._empty_text = arcade.Text(
            "немає сумісних моделей у logs/models", x + 12, y - 46, theme.TEXT_FAINT,
            font_size=11, anchor_x="left", anchor_y="top", font_name=theme.FONT,
        )
        self._row_checkbox_texts: list[arcade.Text] = []
        self._row_name_texts: list[arcade.Text] = []
        self._row_meta_texts: list[arcade.Text] = []
        self._row_count_texts: list[arcade.Text] = []
        # (path_str, left, right, bottom, top, minus_zone, plus_zone) — зони
        # −/+ це (left, right) по X, актуальні лише коли модель обрана.
        self._row_bounds: list[tuple[str, float, float, float, float, tuple[float, float], tuple[float, float]]] = []

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
            self._row_meta_texts.append(arcade.Text(
                "", 0, 0, theme.TEXT_FAINT, font_size=9,
                anchor_x="left", anchor_y="center", font_name=theme.FONT,
            ))
            self._row_count_texts.append(arcade.Text(
                "", 0, 0, theme.TEXT_SECONDARY, font_size=11.5,
                anchor_x="center", anchor_y="center", font_name=theme.FONT,
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
            count = self.counts.get(path_str, 0)

            # Зони −/+ праворуч: [−] [N] [+], кожна 18px.
            plus_zone = (header_right - 24, header_right - 6)
            minus_zone = (header_right - 66, header_right - 48)
            if row_valid:
                self._row_bounds.append((path_str, header_left, header_right, row_bottom, row_top, minus_zone, plus_zone))

            if count > 0 and row_valid:
                arcade.draw_lrbt_rectangle_filled(header_left, header_right, row_bottom, row_top, theme.ACCENT_SOFT)

            checkbox_text = self._row_checkbox_texts[i]
            name_text = self._row_name_texts[i]
            meta_text = self._row_meta_texts[i]
            count_text = self._row_count_texts[i]

            top_row_y = row_y + ROW_HEIGHT / 4
            bottom_row_y = row_y - ROW_HEIGHT / 4

            box_cx = header_left + 18
            box_h = 7.0
            arcade.draw_lrbt_rectangle_outline(
                box_cx - box_h, box_cx + box_h, top_row_y - box_h, top_row_y + box_h,
                theme.ACCENT_STRONG if count > 0 else theme.BORDER, border_width=1.5,
            )
            checkbox_text.text = "✓" if count > 0 else ""
            checkbox_text.x = box_cx
            checkbox_text.y = top_row_y

            name_text.text = path.stem
            name_text.x = header_left + 34
            name_text.y = top_row_y

            steps, _ = read_model_meta(path)
            meta_text.text = _format_steps(steps)
            meta_text.x = header_left + 34
            meta_text.y = bottom_row_y

            if count > 0:
                # [−] N [+]
                for zone, symbol in ((minus_zone, "-"), (plus_zone, "+")):
                    z_left, z_right = zone
                    arcade.draw_lrbt_rectangle_filled(z_left, z_right, top_row_y - 9, top_row_y + 9, theme.BG_CARD)
                    arcade.draw_lrbt_rectangle_outline(z_left, z_right, top_row_y - 9, top_row_y + 9, theme.BORDER, border_width=1)
                count_text.text = f"{count}"
                count_text.x = (minus_zone[1] + plus_zone[0]) / 2
                count_text.y = top_row_y
                # символи −/+ малюємо лініями (не текстом) — центрування точне
                for (z_left, z_right), is_plus in ((minus_zone, False), (plus_zone, True)):
                    z_cx = (z_left + z_right) / 2
                    arcade.draw_line(z_cx - 4, top_row_y, z_cx + 4, top_row_y, theme.TEXT_SECONDARY, line_width=1.6)
                    if is_plus:
                        arcade.draw_line(z_cx, top_row_y - 4, z_cx, top_row_y + 4, theme.TEXT_SECONDARY, line_width=1.6)

            if list_bottom <= row_y <= list_top:
                checkbox_text.draw()
                name_text.draw()
                meta_text.draw()
                if count > 0:
                    count_text.draw()

    def on_mouse_press(self, x: float, y: float) -> None:
        for path_str, left, right, bottom, top, minus_zone, plus_zone in self._row_bounds:
            if not (left <= x <= right and bottom <= y <= top):
                continue
            count = self.counts.get(path_str, 0)
            if count > 0 and minus_zone[0] <= x <= minus_zone[1]:
                self.counts[path_str] = count - 1
            elif count > 0 and plus_zone[0] <= x <= plus_zone[1]:
                self.counts[path_str] = min(MAX_INSTANCES_PER_MODEL, count + 1)
            else:
                # клік по решті рядка — перемикач 0 <-> 1
                self.counts[path_str] = 0 if count > 0 else 1
            return

    def on_mouse_scroll(self, x: float, y: float, scroll_y: int) -> None:
        header_left, header_right, header_bottom, _ = self._header_bounds()
        if not (header_left <= x <= header_right):
            return
        self.scroll_offset -= scroll_y * ROW_HEIGHT * 1.5

    @property
    def any_selected(self) -> bool:
        return any(n > 0 for n in self.counts.values())

    def selected_paths(self) -> list[Path]:
        """Розгорнутий список: модель з лічильником 3 повторюється тричі."""
        result: list[Path] = []
        for path_str in sorted(self.counts):
            result.extend([Path(path_str)] * self.counts[path_str])
        return result


def _format_race_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:05.2f}"


class RaceInfoPanel:
    """Права панель режиму "Заїзди" (замінює SidePanel навчання): кнопка
    нової траси + живий список учасників — колірна плашка, ім'я, номер кола,
    поточний час / найкращий час. Володар найкращого кола серед усіх
    підсвічується зеленим."""

    ROW_H = 40

    def __init__(self, panel_x: float, panel_width: float, window_height: float):
        self.panel_width = panel_width
        self.new_track_button = Button(0, 0, 10, 34, "Нова траса")
        self._section_title = arcade.Text(
            "ЗАЇЗД", 0, 0, theme.ACCENT, font_size=11,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )
        self._hint_text = arcade.Text(
            "обери моделі зліва і натисни\n\"Почати заїзд\"", 0, 0, theme.TEXT_FAINT, font_size=10.5,
            anchor_x="left", anchor_y="top", font_name=theme.FONT, multiline=True, width=240,
        )
        self._row_name_texts: list[arcade.Text] = []
        self._row_time_texts: list[arcade.Text] = []
        self.relayout(panel_x, window_height)

    def relayout(self, panel_x: float, window_height: float) -> None:
        self.panel_x = panel_x
        self.window_height = window_height
        pad = 20
        content_x = panel_x
        content_width = self.panel_width - pad * 2
        y = window_height - 40

        self._section_title.x = content_x - content_width / 2
        self._section_title.y = y
        y -= 34
        self.new_track_button.width = content_width
        self.new_track_button.move(content_x, y)
        y -= 44
        self._list_top = y
        self._content_x = content_x
        self._content_width = content_width
        self._hint_text.x = content_x - content_width / 2
        self._hint_text.y = y

    def _ensure_row_pool(self, n: int) -> None:
        while len(self._row_name_texts) < n:
            self._row_name_texts.append(arcade.Text(
                "", 0, 0, theme.TEXT_PRIMARY, font_size=12,
                anchor_x="left", anchor_y="center", font_name=theme.FONT,
            ))
            self._row_time_texts.append(arcade.Text(
                "", 0, 0, theme.TEXT_FAINT, font_size=9.5,
                anchor_x="left", anchor_y="center", font_name=theme.FONT,
            ))

    def draw(self, session: "RaceSession | None") -> None:
        left = self.panel_x - self.panel_width / 2
        right = self.panel_x + self.panel_width / 2
        arcade.draw_lrbt_rectangle_filled(left, right, 0, self.window_height, theme.BG_RAISED)
        arcade.draw_line(left, 0, left, self.window_height, theme.BORDER_SOFT, line_width=1)

        self._section_title.draw()
        self.new_track_button.draw()

        if session is None or not session.slots:
            self._hint_text.draw()
            return

        self._ensure_row_pool(len(session.slots))
        best_overall = min(
            (s.best_lap_time for s in session.slots if s.best_lap_time is not None),
            default=None,
        )

        row_left = self._content_x - self._content_width / 2
        y = self._list_top
        for i, slot in enumerate(session.slots):
            # Колірна плашка машини
            arcade.draw_lrbt_rectangle_filled(row_left, row_left + 14, y - 7, y + 7, slot.color)
            arcade.draw_lrbt_rectangle_outline(row_left, row_left + 14, y - 7, y + 7, theme.BG, border_width=1)

            is_best = best_overall is not None and slot.best_lap_time == best_overall
            name_text = self._row_name_texts[i]
            name_text.text = slot.name
            name_text.color = theme.GOOD if is_best else theme.TEXT_PRIMARY
            name_text.x = row_left + 22
            name_text.y = y
            name_text.draw()

            time_text = self._row_time_texts[i]
            time_text.text = (
                f"коло №{slot.env.laps_completed + 1}   "
                f"{_format_race_time(slot.env.episode_time)} / {_format_race_time(slot.best_lap_time)}"
            )
            time_text.color = theme.GOOD if is_best else theme.TEXT_FAINT
            time_text.x = row_left + 22
            time_text.y = y - 15
            time_text.draw()

            y -= self.ROW_H

    def on_mouse_motion(self, x: float, y: float) -> None:
        self.new_track_button.hovered = self.new_track_button.contains(x, y)

    def on_mouse_press(self, x: float, y: float) -> str | None:
        if self.new_track_button.contains(x, y):
            return "new_track"
        return None
