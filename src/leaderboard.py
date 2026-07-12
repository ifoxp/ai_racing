"""Таблиця лідерів у верхньому лівому куті viewport. На Етапі 2 показує лише
гравця, пізніше (Етап 6-7) сюди додаються рядки ботів, відсортовані за
найкращим часом кола."""

from __future__ import annotations

import arcade

import theme


def format_lap_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:05.2f}"


class Leaderboard:
    HEIGHT = 100  # 78 (гравець) + 22 для рядка найкращого часу бота

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.width = 220

        self._title_text = arcade.Text(
            "ЛІДЕРИ", x, y, theme.ACCENT, font_size=11,
            anchor_x="left", anchor_y="top", font_name=theme.FONT,
        )
        self._player_name_text = arcade.Text(
            "Гравець", x + 12, y - 34, theme.TEXT_PRIMARY, font_size=13,
            anchor_x="left", anchor_y="top", font_name=theme.FONT, bold=True,
        )
        self._player_time_text = arcade.Text(
            "—", x + self.width - 12, y - 34, theme.TEXT_SECONDARY, font_size=13,
            anchor_x="right", anchor_y="top", font_name=theme.FONT,
        )
        self._current_lap_text = arcade.Text(
            "поточне коло: —", x + 12, y - 56, theme.TEXT_FAINT, font_size=11,
            anchor_x="left", anchor_y="top", font_name=theme.FONT,
        )
        self._bot_name_text = arcade.Text(
            "Бот (найкращий)", x + 12, y - 78, theme.CYAN, font_size=12,
            anchor_x="left", anchor_y="top", font_name=theme.FONT,
        )
        self._bot_time_text = arcade.Text(
            "—", x + self.width - 12, y - 78, theme.TEXT_SECONDARY, font_size=12,
            anchor_x="right", anchor_y="top", font_name=theme.FONT,
        )

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._title_text.x = x
        self._title_text.y = y
        self._player_name_text.x = x + 12
        self._player_name_text.y = y - 34
        self._player_time_text.x = x + self.width - 12
        self._player_time_text.y = y - 34
        self._current_lap_text.x = x + 12
        self._current_lap_text.y = y - 56
        self._bot_name_text.x = x + 12
        self._bot_name_text.y = y - 78
        self._bot_time_text.x = x + self.width - 12
        self._bot_time_text.y = y - 78

    def draw(self, best_time: float | None, current_lap_time: float | None,
              bot_best_time: float | None = None) -> None:
        height = self.HEIGHT
        left, right = self.x, self.x + self.width
        top, bottom = self.y, self.y - height

        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, theme.BG_RAISED + (215,))
        arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, theme.BORDER_SOFT, border_width=1.5)

        self._title_text.draw()
        self._player_name_text.draw()

        self._player_time_text.text = format_lap_time(best_time)
        self._player_time_text.draw()

        if current_lap_time is not None:
            self._current_lap_text.text = f"поточне коло: {format_lap_time(current_lap_time)}"
        else:
            self._current_lap_text.text = "поточне коло: —"
        self._current_lap_text.draw()

        self._bot_name_text.draw()
        self._bot_time_text.text = format_lap_time(bot_best_time)
        self._bot_time_text.draw()
