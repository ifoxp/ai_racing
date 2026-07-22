"""Вкладка "Налаштування" — редагування всіх числових параметрів нагород і
кутів променів прямо з гри, без правки коду.

Права панель: поля вводу (кожне з кнопкою скидання до дефолту), чекбокс
"виліт = смерть", кнопка "Застосувати". Зміни летять у ЖИВІ підпроцеси
навчання через trainer.update_reward_config() (queued env_method) — не треба
перезапускати тренування, щоб поекспериментувати з цифрами.

Лівіше, у viewport — прев'ю: машинка з віялом променів під поточними кутами
(читаються з полів наживо, ще ДО натискання "Застосувати") — одразу видно,
куди дивитимуться сенсори.
"""

from __future__ import annotations

from dataclasses import replace

import arcade
import arcade.gui
import numpy as np

import theme
from car import CAR_LENGTH, CAR_WIDTH
from settings import DEFAULT_RAY_HALF_ANGLES, RewardConfig
from ui_panel import Button, Checkbox, IconButton, Section

ROW_H = 34
INPUT_W = 74
INPUT_H = 24
RESET_SIZE = 22

# (attr, підпис, дефолт) — числові поля у порядку відображення.
_NUMBER_FIELDS: list[tuple[str, str, float]] = [
    ("checkpoint_bonus", "Бонус за checkpoint", RewardConfig().checkpoint_bonus),
    ("lap_bonus_scale", "Бонус за коло (множник)", RewardConfig().lap_bonus_scale),
    ("speed_reward_scale", "Швидкість (×v щокадру)", RewardConfig().speed_reward_scale),
    ("brake_bonus_scale", "Гальмо в повороті (×v)", RewardConfig().brake_bonus_scale),
    ("stuck_penalty", "Штраф застрягання (щокадру)", RewardConfig().stuck_penalty),
    ("out_of_bounds_penalty", "Штраф за виліт", RewardConfig().out_of_bounds_penalty),
    ("score_death_threshold", "Смерть при рахунку нижче", RewardConfig().score_death_threshold),
]


def _fmt(value: float) -> str:
    """Число без хвостових нулів: 1.0 → "1", 0.002 → "0.002"."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


class _FieldRow:
    """Один рядок: підпис + поле вводу + кнопка скидання до дефолту."""

    def __init__(self, ui_manager: arcade.gui.UIManager, label: str, default: float):
        self.default = default
        self.label_text = arcade.Text(
            label, 0, 0, theme.TEXT_SECONDARY, font_size=10.5,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )
        self.input = arcade.gui.UIInputText(
            x=0, y=0, width=INPUT_W, height=INPUT_H, text=_fmt(default), font_size=11,
            text_color=theme.TEXT_PRIMARY, caret_color=theme.ACCENT, border_color=theme.BORDER,
        ).with_background(color=arcade.types.Color(*theme.BG_CARD)).with_padding(left=6, top=4)
        ui_manager.add(self.input)
        self.reset_btn = IconButton(0, 0, RESET_SIZE, "reset")

    def move(self, left: float, right: float, y: float) -> None:
        self.label_text.x = left
        self.label_text.y = y
        self.reset_btn.move(right - RESET_SIZE / 2, y)
        input_right = right - RESET_SIZE - 6
        self.input.rect = self.input.rect.align_left(input_right - INPUT_W).align_top(y + INPUT_H / 2)

    def value(self, fallback: float) -> float:
        """Парсить поле; кома як роздільник теж приймається. Нечитабельний
        текст не ламає застосування — повертається fallback (поточне
        значення конфіга), а поле повертається до нього ж."""
        try:
            return float(self.input.text.strip().replace(",", "."))
        except ValueError:
            self.input.text = _fmt(fallback)
            return fallback

    def set_value(self, value: float) -> None:
        self.input.text = _fmt(value)

    def draw(self) -> None:
        self.label_text.draw()
        self.reset_btn.draw()

    def on_mouse_press(self, x: float, y: float) -> bool:
        if self.reset_btn.contains(x, y):
            self.set_value(self.default)
            self.reset_btn.trigger_flash()
            return True
        return False


class SettingsPanel:
    """Права панель вкладки "Налаштування" + прев'ю променів у viewport."""

    def __init__(self, window: arcade.Window, panel_x: float, panel_width: float, window_height: float,
                 config: RewardConfig):
        self.window = window
        self.panel_width = panel_width
        self.config = config
        self.ui_manager = arcade.gui.UIManager(window)
        # НЕ enable() тут — вмикається лише коли вкладка активна (main.py),
        # інакше невидимі поля перехоплювали б кліки в інших вкладках.

        self.sections: list[Section] = []
        self.reward_rows: list[_FieldRow] = [
            _FieldRow(self.ui_manager, label, default) for _, label, default in _NUMBER_FIELDS
        ]
        self.death_checkbox = Checkbox(0, 0, 18, "виліт = смерть епізоду", checked=config.out_of_bounds_is_death)
        self.ray_rows: list[_FieldRow] = [
            _FieldRow(self.ui_manager, f"Промінь {i + 1} (±°)", DEFAULT_RAY_HALF_ANGLES[i])
            for i in range(len(DEFAULT_RAY_HALF_ANGLES))
        ]
        self.apply_button = Button(0, 0, 10, 34, "Застосувати")
        self._hint_text = arcade.Text(
            "діє одразу, навіть під час навчання", 0, 0, theme.TEXT_FAINT, font_size=9.5,
            anchor_x="center", anchor_y="center", font_name=theme.FONT,
        )
        self._preview_angle_texts: list[arcade.Text] = [
            arcade.Text("", 0, 0, theme.TEXT_FAINT, font_size=10,
                        anchor_x="center", anchor_y="center", font_name=theme.FONT)
            for _ in range(len(DEFAULT_RAY_HALF_ANGLES) * 2)
        ]

        self.set_config(config)
        self.relayout(panel_x, window_height)

    # --- Конфіг <-> поля ---

    def set_config(self, config: RewardConfig) -> None:
        self.config = config
        for row, (attr, _, _) in zip(self.reward_rows, _NUMBER_FIELDS):
            row.set_value(getattr(config, attr))
        self.death_checkbox.checked = config.out_of_bounds_is_death
        for row, angle in zip(self.ray_rows, config.ray_half_angles):
            row.set_value(angle)

    def read_config(self) -> RewardConfig:
        """Збирає RewardConfig з полів. Кути затискаються в [1°, 179°] —
        0° зламав би симетрію ± (два однакові промені), а ≥180° дзеркалить
        назад у той самий діапазон, лише заплутуючи."""
        values = {
            attr: row.value(getattr(self.config, attr))
            for row, (attr, _, _) in zip(self.reward_rows, _NUMBER_FIELDS)
        }
        angles = [
            float(np.clip(row.value(self.config.ray_half_angles[i]), 1.0, 179.0))
            for i, row in enumerate(self.ray_rows)
        ]
        return replace(
            self.config, **values,
            out_of_bounds_is_death=self.death_checkbox.checked,
            ray_half_angles=angles,
        )

    def _preview_angles(self) -> list[float]:
        """Кути для прев'ю — з полів наживо (ще до "Застосувати")."""
        return [
            float(np.clip(row.value(self.config.ray_half_angles[i]), 1.0, 179.0))
            for i, row in enumerate(self.ray_rows)
        ]

    # --- Розкладка/рендер ---

    def relayout(self, panel_x: float, window_height: float) -> None:
        self.panel_x = panel_x
        self.window_height = window_height
        pad = 20
        content_x = panel_x
        content_width = self.panel_width - pad * 2
        left = content_x - content_width / 2
        right = content_x + content_width / 2
        y = window_height - 40

        first_pass = not self.sections
        if first_pass:
            self.sections.append(Section(content_x, y, content_width, "НАГОРОДИ"))
        else:
            self.sections[0].move(content_x, y)
        y -= 32

        for row in self.reward_rows:
            row.move(left, right, y)
            y -= ROW_H

        self.death_checkbox.move(left + 9, y)
        y -= 40

        if first_pass:
            self.sections.append(Section(content_x, y, content_width, "ПРОМЕНІ"))
        else:
            self.sections[1].move(content_x, y)
        y -= 32

        for row in self.ray_rows:
            row.move(left, right, y)
            y -= ROW_H

        y -= 8
        self.apply_button.width = content_width
        self.apply_button.move(content_x, y)
        self._hint_text.x = content_x
        self._hint_text.y = y - 26

    def draw(self) -> None:
        left = self.panel_x - self.panel_width / 2
        right = self.panel_x + self.panel_width / 2
        arcade.draw_lrbt_rectangle_filled(left, right, 0, self.window_height, theme.BG_RAISED)
        arcade.draw_line(left, 0, left, self.window_height, theme.BORDER_SOFT, line_width=1)

        for section in self.sections:
            section.draw()
        for row in self.reward_rows + self.ray_rows:
            row.draw()
        self.death_checkbox.draw()
        self.apply_button.draw()
        self._hint_text.draw()
        self.ui_manager.draw()

    def draw_preview(self, cx: float, cy: float) -> None:
        """Машинка з віялом променів у центрі viewport — кути беруться з
        полів наживо, тож редагування видно одразу."""
        scale = 5.0
        ray_len = 230.0
        half_angles = self._preview_angles()
        full = sorted([-a for a in half_angles] + half_angles)

        text_idx = 0
        for angle_deg in full:
            angle = np.radians(angle_deg)
            ex = cx + np.cos(angle) * ray_len
            ey = cy + np.sin(angle) * ray_len
            arcade.draw_line(cx, cy, ex, ey, theme.CYAN + (140,), line_width=1.4)
            label = self._preview_angle_texts[text_idx]
            label.text = f"{angle_deg:g}°"
            label.x = cx + np.cos(angle) * (ray_len + 22)
            label.y = cy + np.sin(angle) * (ray_len + 22)
            label.draw()
            text_idx += 1

        # Корпус машини (реальні пропорції CAR_LENGTH×CAR_WIDTH), ніс праворуч
        # (0° = +X) — так само, як heading у фізиці.
        half_l = CAR_LENGTH / 2 * scale
        half_w = CAR_WIDTH / 2 * scale
        arcade.draw_lrbt_rectangle_filled(cx - half_l, cx + half_l, cy - half_w, cy + half_w, theme.ACCENT_STRONG)
        arcade.draw_lrbt_rectangle_outline(cx - half_l, cx + half_l, cy - half_w, cy + half_w, theme.BG, border_width=2)
        arcade.draw_triangle_filled(
            cx + half_l, cy + half_w * 0.6, cx + half_l, cy - half_w * 0.6, cx + half_l + 14, cy, theme.BG,
        )

    # --- Події ---

    @property
    def any_input_active(self) -> bool:
        """Хоч одне текстове поле в фокусі — головне вікно тоді не має
        реагувати на R/WASD (користувач друкує цифри, не керує грою)."""
        return any(row.input.active for row in self.reward_rows + self.ray_rows)

    def on_update(self, delta_time: float) -> None:
        for row in self.reward_rows + self.ray_rows:
            row.reset_btn.update(delta_time)

    def on_mouse_motion(self, x: float, y: float) -> None:
        self.apply_button.hovered = self.apply_button.contains(x, y)
        self.death_checkbox.hovered = self.death_checkbox.contains(x, y)
        for row in self.reward_rows + self.ray_rows:
            row.reset_btn.hovered = row.reset_btn.contains(x, y)

    def on_mouse_press(self, x: float, y: float) -> str | None:
        for row in self.reward_rows + self.ray_rows:
            if row.on_mouse_press(x, y):
                return None
        if self.death_checkbox.contains(x, y):
            self.death_checkbox.checked = not self.death_checkbox.checked
            return None
        if self.apply_button.contains(x, y):
            return "apply"
        return None
