"""Права панель UI — каркас Етапу 1.

Більшість елементів тут — навмисно неактивні заглушки (вимога Етапу 1: закласти
місце під майбутній функціонал, не реалізовувати його одразу). Активна на цьому
етапі лише секція генерації траси.

Текст малюється через arcade.Text (об'єкт створюється один раз, тільки .draw()
у циклі рендеру) — arcade.draw_text() у циклі on_draw офіційно позначений як
повільний і не рекомендований для щокадрового виклику.
"""

from __future__ import annotations

import arcade

import theme


class Button:
    def __init__(self, x: float, y: float, width: float, height: float,
                 label: str, enabled: bool = True):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.enabled = enabled
        self.hovered = False
        self._text = arcade.Text(
            label, x, y, theme.TEXT_FAINT, font_size=12,
            anchor_x="center", anchor_y="center", font_name=theme.FONT,
        )

    def contains(self, x: float, y: float) -> bool:
        return (self.x - self.width / 2 <= x <= self.x + self.width / 2
                and self.y - self.height / 2 <= y <= self.y + self.height / 2)

    def draw(self) -> None:
        left = self.x - self.width / 2
        right = self.x + self.width / 2
        bottom = self.y - self.height / 2
        top = self.y + self.height / 2

        if not self.enabled:
            fill = theme.BG_RAISED
            border = theme.BORDER_SOFT
            text_color = theme.TEXT_FAINT
        elif self.hovered:
            fill = theme.BG_CARD
            border = theme.ACCENT
            text_color = theme.ACCENT_STRONG
        else:
            fill = theme.BG_CARD
            border = theme.BORDER
            text_color = theme.TEXT_SECONDARY

        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, fill)
        arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, border, border_width=1.5)
        self._text.color = text_color
        self._text.draw()


class Section:
    """Заголовок-розділювач всередині панелі (напр. 'ТРАСА', 'НАВЧАННЯ')."""

    def __init__(self, x: float, y: float, width: float, label: str):
        self.x = x
        self.y = y
        self.width = width
        self.label = label
        self._text = arcade.Text(
            label, x - width / 2, y, theme.ACCENT, font_size=11,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )

    def draw(self) -> None:
        self._text.draw()
        arcade.draw_line(
            self.x - self.width / 2, self.y - 12, self.x + self.width / 2, self.y - 12,
            theme.BORDER_SOFT, line_width=1,
        )


class SidePanel:
    """Права панель — трек-контроли (активні) + заглушки майбутніх етапів."""

    def __init__(self, panel_x: float, panel_width: float, window_height: float):
        self.panel_x = panel_x
        self.panel_width = panel_width
        self.window_height = window_height

        pad = 20
        content_x = panel_x
        content_width = panel_width - pad * 2
        y = window_height - 40

        self.sections: list[Section] = []
        self.buttons: dict[str, Button] = {}

        # --- Секція: ТРАСА (активна на Етапі 1) ---
        self.sections.append(Section(content_x, y, content_width, "ТРАСА"))
        y -= 34
        self.buttons["new_track"] = Button(content_x, y, content_width, 34, "Нова траса", enabled=True)
        y -= 42
        self.buttons["same_seed"] = Button(content_x, y, content_width, 34, "Той самий seed", enabled=True)
        y -= 56

        # --- Секція: КЕРУВАННЯ (Етап 2, заглушка) ---
        self.sections.append(Section(content_x, y, content_width, "КЕРУВАННЯ"))
        y -= 34
        self.buttons["manual_drive"] = Button(content_x, y, content_width, 34, "Ручна їзда (WASD)", enabled=False)
        y -= 42
        self.buttons["lap_timer"] = Button(content_x, y, content_width, 34, "Таймер кола", enabled=False)
        y -= 56

        # --- Секція: ЗАПИС (Етап 3, заглушка) ---
        self.sections.append(Section(content_x, y, content_width, "ЗАПИС ЗАЇЗДУ"))
        y -= 34
        self.buttons["record"] = Button(content_x, y, content_width, 34, "Записати заїзд", enabled=False)
        y -= 42
        self.buttons["ghost"] = Button(content_x, y, content_width, 34, "Показати ghost", enabled=False)
        y -= 56

        # --- Секція: НАВЧАННЯ (Етап 4, заглушка) ---
        self.sections.append(Section(content_x, y, content_width, "НАВЧАННЯ"))
        y -= 34
        self.buttons["train_start"] = Button(content_x, y, content_width, 34, "Почати навчання", enabled=False)
        y -= 42
        self.buttons["top1"] = Button(content_x - content_width / 3, y, content_width / 3 - 4, 30, "Топ-1", enabled=False)
        self.buttons["top10"] = Button(content_x, y, content_width / 3 - 4, 30, "Топ-10", enabled=False)
        self.buttons["top100"] = Button(content_x + content_width / 3, y, content_width / 3 - 4, 30, "Топ-100", enabled=False)
        y -= 42
        self.buttons["reward_reset"] = Button(content_x, y, content_width, 30, "Reward: скинути", enabled=False)
        y -= 56

        # --- Секція: ПОКОЛІННЯ (Етап 5, заглушка) ---
        self.sections.append(Section(content_x, y, content_width, "ПОКОЛІННЯ"))
        y -= 34
        self.buttons["save_gen"] = Button(content_x - content_width / 4, y, content_width / 2 - 4, 30, "Зберегти", enabled=False)
        self.buttons["load_gen"] = Button(content_x + content_width / 4, y, content_width / 2 - 4, 30, "Завантажити", enabled=False)
        y -= 56

        self.info_y_bottom = y

        info_x = content_x - (self.panel_width - pad * 2) / 2
        self._seed_text = arcade.Text(
            "", info_x, self.info_y_bottom, theme.TEXT_FAINT, font_size=11,
            anchor_x="left", font_name=theme.FONT,
        )
        self._gen_time_text = arcade.Text(
            "", info_x, self.info_y_bottom - 18, theme.TEXT_FAINT, font_size=11,
            anchor_x="left", font_name=theme.FONT,
        )

    def draw(self, seed: int | None = None, track_ms: float | None = None) -> None:
        left = self.panel_x - self.panel_width / 2
        right = self.panel_x + self.panel_width / 2
        arcade.draw_lrbt_rectangle_filled(left, right, 0, self.window_height, theme.BG_RAISED)
        arcade.draw_line(left, 0, left, self.window_height, theme.BORDER_SOFT, line_width=1)

        for section in self.sections:
            section.draw()
        for button in self.buttons.values():
            button.draw()

        if seed is not None:
            self._seed_text.text = f"seed: {seed}"
            self._seed_text.draw()
        if track_ms is not None:
            self._gen_time_text.text = f"генерація: {track_ms:.1f} мс"
            self._gen_time_text.draw()

    def on_mouse_motion(self, x: float, y: float) -> None:
        for button in self.buttons.values():
            button.hovered = button.enabled and button.contains(x, y)

    def on_mouse_press(self, x: float, y: float) -> str | None:
        for name, button in self.buttons.items():
            if button.enabled and button.contains(x, y):
                return name
        return None
