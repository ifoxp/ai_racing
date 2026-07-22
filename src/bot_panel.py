"""Панель ботів у лівому верхньому куті viewport, під Leaderboard гравця —
список усіх ботів, що зараз тренуються, відсортований за поточним reward
(найкращий зверху). Згортається/розгортається кліком по заголовку, кількість
видимих рядків обирається випадаючим списком (1/5/10/25/50/100), решта —
під скролом (коліщатко миші). Клік по рядку виділяє бота — main.py
підсвічує саме його на трасі.
"""

from __future__ import annotations

import arcade

import theme

ROW_HEIGHT = 38  # основний рядок (ім'я+бали) + компактний рядок часів кола під ним
MAX_LIST_HEIGHT = 260  # видима висота списку (скрол понад це) у пікселях


def _visible_count_options(bot_count: int) -> list[int]:
    """Варіанти "скільки ботів показувати" залежать від того, скільки їх
    реально запущено — фіксований список (1/5/10/25/50/100) для 8 ботів
    пропонував безглузді 25/50/100. Тепер: 1, чверть, половина, всі
    (для 8 → 1/2/4/8, для 16 → 1/4/8/16, для 24 → 1/6/12/24)."""
    if bot_count <= 1:
        return [1]
    return sorted({1, max(1, round(bot_count / 4)), max(1, round(bot_count / 2)), bot_count})


def _format_lap_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:05.2f}"


class BotPanel:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.width = 220
        self.collapsed = False
        self.options = [1]  # перераховуються в draw() від реальної кількості ботів
        self.visible_count_idx = 0
        self.scroll_offset = 0.0
        self.selected_bot_id: int | None = None

        self._title_text = arcade.Text(
            "БОТИ", x + 12, y - 18, theme.ACCENT, font_size=11,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )
        self._collapse_hint_text = arcade.Text(
            "▾", x, y - 18, theme.TEXT_FAINT, font_size=13,
            anchor_x="right", anchor_y="center", font_name=theme.FONT,
        )
        self._count_label_text = arcade.Text(
            "", x, y - 18, theme.TEXT_SECONDARY, font_size=11,
            anchor_x="right", anchor_y="center", font_name=theme.FONT,
        )
        self._empty_text = arcade.Text(
            "навчання не запущено", x + 12, y - 46, theme.TEXT_FAINT, font_size=11,
            anchor_x="left", anchor_y="top", font_name=theme.FONT,
        )
        self._row_texts: list[tuple[arcade.Text, arcade.Text, arcade.Text]] = []  # (name, reward, lap_info) — пул, що переростає
        self._row_bounds: list[tuple[int, float, float, float, float]] = []  # (bot_id, left, right, bottom, top) для кліків

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._reposition_static()

    def _reposition_static(self) -> None:
        self._title_text.x = self.x + 12
        self._title_text.y = self.y - 18
        self._collapse_hint_text.x = self.x + self.width - 12
        self._collapse_hint_text.y = self.y - 18
        self._count_label_text.x = self.x + self.width - 26
        self._count_label_text.y = self.y - 18
        self._empty_text.x = self.x + 12
        self._empty_text.y = self.y - 46

    def _header_bounds(self) -> tuple[float, float, float, float]:
        return self.x, self.x + self.width, self.y - 30, self.y

    def _dropdown_bounds(self) -> tuple[float, float, float, float]:
        return self.x + self.width - 52, self.x + self.width, self.y - 30, self.y

    def _ensure_row_pool(self, n: int) -> None:
        while len(self._row_texts) < n:
            name_text = arcade.Text(
                "", 0, 0, theme.TEXT_PRIMARY, font_size=12,
                anchor_x="left", anchor_y="center", font_name=theme.FONT,
            )
            reward_text = arcade.Text(
                "", 0, 0, theme.TEXT_SECONDARY, font_size=12,
                anchor_x="right", anchor_y="center", font_name=theme.FONT,
            )
            lap_info_text = arcade.Text(
                "", 0, 0, theme.TEXT_FAINT, font_size=9.5,
                anchor_x="left", anchor_y="center", font_name=theme.FONT,
            )
            self._row_texts.append((name_text, reward_text, lap_info_text))

    def draw(self, bots: list[tuple[int, str, float, float | None, float, int]], total_timesteps: int = 0) -> None:
        """bots: список (bot_id, name, episode_reward, best_lap_time,
        current_lap_time, lap_number), НЕ обов'язково відсортований —
        сортування й обрізання до visible_count тут.
        total_timesteps: загальний прогрес моделі за весь час її життя (не
        обнуляється при завантаженні збереженої моделі, на відміну від
        episode_reward у списку — той завжди рахує лише поточний епізод)."""
        header_left, header_right, header_bottom, header_top = self._header_bounds()
        arcade.draw_lrbt_rectangle_filled(header_left, header_right, header_bottom, header_top, theme.BG_RAISED + (215,))
        arcade.draw_lrbt_rectangle_outline(header_left, header_right, header_bottom, header_top, theme.BORDER_SOFT, border_width=1.5)
        if total_timesteps > 0:
            steps_label = f"{total_timesteps / 1000:.1f}K" if total_timesteps >= 1000 else str(total_timesteps)
            self._title_text.text = f"БОТИ · {steps_label} кроків"
        else:
            self._title_text.text = "БОТИ"
        self._title_text.draw()
        self._collapse_hint_text.text = "▸" if self.collapsed else "▾"
        self._collapse_hint_text.draw()

        # Варіанти залежать від реальної кількості ботів — при зміні списку
        # (нове навчання з іншим bot_count) зберігаємо не індекс, а найближче
        # ЗНАЧЕННЯ до поточного вибору (щоб "показувати 8" не стало раптом
        # "показувати 1" лише тому, що список перебудувався).
        new_options = _visible_count_options(len(bots)) if bots else [1]
        if new_options != self.options:
            current_value = self.options[min(self.visible_count_idx, len(self.options) - 1)]
            self.options = new_options
            self.visible_count_idx = min(
                range(len(new_options)), key=lambda i: abs(new_options[i] - current_value)
            )

        visible_count = self.options[self.visible_count_idx]
        self._count_label_text.text = str(visible_count)
        self._count_label_text.draw()

        self._row_bounds = []

        if self.collapsed:
            return

        if not bots:
            list_bottom = header_bottom - 34
            arcade.draw_lrbt_rectangle_filled(header_left, header_right, list_bottom, header_bottom, theme.BG_RAISED + (215,))
            arcade.draw_lrbt_rectangle_outline(header_left, header_right, list_bottom, header_bottom, theme.BORDER_SOFT, border_width=1.5)
            self._empty_text.draw()
            return

        ranked = sorted(bots, key=lambda b: b[2], reverse=True)[:visible_count]
        list_height = min(len(ranked) * ROW_HEIGHT, MAX_LIST_HEIGHT)
        max_scroll = max(0.0, len(ranked) * ROW_HEIGHT - MAX_LIST_HEIGHT)
        self.scroll_offset = max(0.0, min(self.scroll_offset, max_scroll))

        list_top = header_bottom
        list_bottom = list_top - list_height
        arcade.draw_lrbt_rectangle_filled(header_left, header_right, list_bottom, list_top, theme.BG_RAISED + (215,))
        arcade.draw_lrbt_rectangle_outline(header_left, header_right, list_bottom, list_top, theme.BORDER_SOFT, border_width=1.5)

        self._ensure_row_pool(len(ranked))

        for i, (bot_id, name, reward, best_lap, current_lap, lap_number) in enumerate(ranked):
            row_y = list_top - i * ROW_HEIGHT + self.scroll_offset - ROW_HEIGHT / 2
            if row_y > list_top or row_y < list_bottom - ROW_HEIGHT:
                continue  # рядок повністю за межами видимої області — не малюємо (і не оновлюємо текст)

            row_top = min(row_y + ROW_HEIGHT / 2, list_top)
            row_bottom = max(row_y - ROW_HEIGHT / 2, list_bottom)
            row_valid = row_bottom < row_top
            if row_valid:
                self._row_bounds.append((bot_id, header_left, header_right, row_bottom, row_top))

            is_selected = bot_id == self.selected_bot_id
            if is_selected and row_valid:
                arcade.draw_lrbt_rectangle_filled(header_left, header_right, row_bottom, row_top, theme.ACCENT_SOFT)

            name_text, reward_text, lap_info_text = self._row_texts[i]
            if is_selected:
                rank_color = theme.ACCENT_STRONG
            elif i == 0:
                rank_color = theme.GOOD
            else:
                rank_color = theme.TEXT_PRIMARY

            # Верхній піврядок — ім'я + поточний reward епізоду.
            top_row_y = row_y + ROW_HEIGHT / 4
            name_text.text = f"{i + 1}. {name}"
            name_text.color = rank_color
            name_text.x = header_left + 12
            name_text.y = top_row_y
            reward_text.text = f"{reward:.1f}"
            reward_text.x = header_right - 12
            reward_text.y = top_row_y

            # Нижній піврядок — компактно: коло N, поточний час / найкращий час.
            bottom_row_y = row_y - ROW_HEIGHT / 4
            lap_info_text.text = (
                f"№{lap_number}  {_format_lap_time(current_lap)} / {_format_lap_time(best_lap)}"
            )
            lap_info_text.x = header_left + 12
            lap_info_text.y = bottom_row_y

            if list_bottom <= row_y <= list_top:
                name_text.draw()
                reward_text.draw()
                lap_info_text.draw()

    def on_mouse_press(self, x: float, y: float) -> None:
        dd_left, dd_right, dd_bottom, dd_top = self._dropdown_bounds()
        if dd_left <= x <= dd_right and dd_bottom <= y <= dd_top:
            self.visible_count_idx = (self.visible_count_idx + 1) % len(self.options)
            return

        header_left, header_right, header_bottom, header_top = self._header_bounds()
        if header_left <= x <= header_right and header_bottom <= y <= header_top:
            self.collapsed = not self.collapsed
            return

        for bot_id, left, right, bottom, top in self._row_bounds:
            if left <= x <= right and bottom <= y <= top:
                self.selected_bot_id = None if self.selected_bot_id == bot_id else bot_id
                return

    @property
    def visible_count(self) -> int:
        return self.options[min(self.visible_count_idx, len(self.options) - 1)]

    def on_mouse_scroll(self, x: float, y: float, scroll_y: int) -> None:
        if self.collapsed:
            return
        header_left, header_right, header_bottom, _ = self._header_bounds()
        if not (header_left <= x <= header_right):
            return
        self.scroll_offset -= scroll_y * ROW_HEIGHT * 1.5


class ControlsIndicator:
    """Іконки газу/гальма/керма в нижньому лівому куті viewport — показує,
    що САМЕ зараз натискає виділений бот (не гравець). Малюється лише коли
    є обраний бот (BotPanel.selected_bot_id не None) — дає змогу візуально
    перевірити гіпотезу типу "бот не хоче гальмувати" в реальному часі,
    дивлячись на конкретного бота, а не вгадуючи з руху машинки."""

    ICON_SIZE = 40
    GAP = 8

    def __init__(self, x: float, y: float):
        self.x = x  # лівий край першої (газ) іконки
        self.y = y  # центр по вертикалі

        self._label_text = arcade.Text(
            "", 0, 0, theme.TEXT_FAINT, font_size=10.5,
            anchor_x="left", anchor_y="bottom", font_name=theme.FONT,
        )
        self._reposition_label()

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._reposition_label()

    def _reposition_label(self) -> None:
        self._label_text.x = self.x
        self._label_text.y = self.y + self.ICON_SIZE / 2 + 6

    def _slot_center(self, index: int) -> tuple[float, float]:
        s = self.ICON_SIZE
        return self.x + s / 2 + index * (s + self.GAP), self.y

    def _draw_slot_bg(self, cx: float, cy: float, active: bool) -> None:
        h = self.ICON_SIZE / 2
        left, right, bottom, top = cx - h, cx + h, cy - h, cy + h
        fill = theme.ACCENT_SOFT if active else theme.BG_CARD
        border = theme.ACCENT_STRONG if active else theme.BORDER
        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, fill)
        arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, border, border_width=1.5)

    def draw(self, name: str | None, throttle: int, steer: float, brake: float) -> None:
        if name is None:
            return

        self._label_text.text = f"керування: {name}"
        self._label_text.draw()

        ink_active, ink_idle = theme.ACCENT_STRONG, theme.TEXT_FAINT

        # 0: гальмо (квадрат-стоп), 1: ліво, 2: газ (трикутник вгору), 3: право
        brake_on = brake > 0.5
        cx, cy = self._slot_center(0)
        self._draw_slot_bg(cx, cy, brake_on)
        s = self.ICON_SIZE * 0.22
        arcade.draw_lrbt_rectangle_filled(cx - s, cx + s, cy - s, cy + s, ink_active if brake_on else ink_idle)

        left_on = steer > 0.15  # steer>0 = проти годинникової (вліво в екранних координатах heading)
        cx, cy = self._slot_center(1)
        self._draw_slot_bg(cx, cy, left_on)
        s = self.ICON_SIZE * 0.24
        ink = ink_active if left_on else ink_idle
        arcade.draw_triangle_filled(cx + s, cy + s, cx + s, cy - s, cx - s, cy, ink)

        throttle_on = throttle > 0
        cx, cy = self._slot_center(2)
        self._draw_slot_bg(cx, cy, throttle_on)
        s = self.ICON_SIZE * 0.24
        ink = ink_active if throttle_on else ink_idle
        arcade.draw_triangle_filled(cx - s, cy - s, cx + s, cy - s, cx, cy + s, ink)

        right_on = steer < -0.15
        cx, cy = self._slot_center(3)
        self._draw_slot_bg(cx, cy, right_on)
        s = self.ICON_SIZE * 0.24
        ink = ink_active if right_on else ink_idle
        arcade.draw_triangle_filled(cx - s, cy + s, cx - s, cy - s, cx + s, cy, ink)
