"""CarAI — Етап 1: генерація траси + каркас UI.

Запуск: venv/Scripts/python.exe src/main.py
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import arcade

import theme
from track_generator import Track, generate_track
from ui_panel import SidePanel

WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900
PANEL_WIDTH = 300
VIEWPORT_WIDTH = WINDOW_WIDTH - PANEL_WIDTH

LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "track_generation"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("carai.track")
logger.setLevel(logging.INFO)
_log_file = LOG_DIR / f"{time.strftime('%Y-%m-%d')}_run.log"
_handler = logging.FileHandler(_log_file, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
if not logger.handlers:
    logger.addHandler(_handler)


class CarAIWindow(arcade.Window):
    def __init__(self) -> None:
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "CarAI", resizable=False)
        arcade.set_background_color(theme.BG)

        self.panel = SidePanel(
            panel_x=VIEWPORT_WIDTH + PANEL_WIDTH / 2,
            panel_width=PANEL_WIDTH,
            window_height=WINDOW_HEIGHT,
        )

        self.track: Track | None = None
        self.seed: int | None = None
        self.last_generation_ms: float | None = None

        self._fps_text = arcade.Text(
            "", VIEWPORT_WIDTH - 16, WINDOW_HEIGHT - 16, theme.TEXT_FAINT,
            font_size=12, anchor_x="right", anchor_y="top", font_name=theme.FONT,
        )
        self._title_text = arcade.Text(
            "CarAI", 24, WINDOW_HEIGHT - 16, theme.TEXT_PRIMARY, font_size=16,
            anchor_x="left", anchor_y="top", font_name=theme.FONT, bold=True,
        )

        # FPS-лічильник — ковзне середнє за останні N кадрів (закладено з Етапу 1,
        # знадобиться повноцінно на Етапі 4 для моніторингу продуктивності навчання)
        self.frame_times: list[float] = []
        self._last_frame_stamp = time.perf_counter()

        self._generate_new_track(seed=random.randint(0, 1_000_000))

    def _generate_new_track(self, seed: int) -> None:
        start = time.perf_counter()
        try:
            track = generate_track(seed=seed)
        except Exception:
            logger.exception("Помилка генерації траси, seed=%s", seed)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.track = track
        self.seed = seed
        self.last_generation_ms = elapsed_ms
        logger.info(
            "seed=%s  точок=%d  час=%.1fмс  checkpoints=%d",
            seed, len(track.control_points), elapsed_ms, len(track.checkpoints),
        )

    def _track_to_screen(self, points):
        """Трек генерується у світових координатах з центром (0,0) — переносимо
        у видиму область viewport з відступами і масштабом."""
        if self.track is None or len(points) == 0:
            return []
        margin = 60
        avail_w = VIEWPORT_WIDTH - margin * 2
        avail_h = WINDOW_HEIGHT - margin * 2

        xs = self.track.left_edge[:, 0].tolist() + self.track.right_edge[:, 0].tolist()
        ys = self.track.left_edge[:, 1].tolist() + self.track.right_edge[:, 1].tolist()
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale = min(avail_w / span_x, avail_h / span_y)

        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        offset_x = VIEWPORT_WIDTH / 2
        offset_y = WINDOW_HEIGHT / 2

        return [
            (offset_x + (p[0] - cx) * scale, offset_y + (p[1] - cy) * scale)
            for p in points
        ]

    def on_draw(self) -> None:
        self.clear()

        # Viewport — трава
        arcade.draw_lrbt_rectangle_filled(0, VIEWPORT_WIDTH, 0, WINDOW_HEIGHT, theme.GRASS)

        if self.track is not None:
            left = self._track_to_screen(self.track.left_edge)
            right = self._track_to_screen(self.track.right_edge)

            # Асфальт: трикутна смуга між left і right контурами
            n = len(left)
            for i in range(n):
                j = (i + 1) % n
                quad = [left[i], left[j], right[j], right[i]]
                arcade.draw_polygon_filled(quad, theme.ASPHALT)

            arcade.draw_line_strip(left + [left[0]], theme.ASPHALT_EDGE, line_width=2)
            arcade.draw_line_strip(right + [right[0]], theme.ASPHALT_EDGE, line_width=2)

            # Checkpoints — маленькі позначки вздовж центральної лінії
            checkpoints_screen = self._track_to_screen(self.track.checkpoints)
            for cx, cy in checkpoints_screen:
                arcade.draw_circle_outline(cx, cy, 5, theme.CYAN, border_width=1.5)

            # Старт — акцентна точка
            start_screen = self._track_to_screen(self.track.center_line[:1])
            if start_screen:
                sx, sy = start_screen[0]
                arcade.draw_circle_filled(sx, sy, 7, theme.ACCENT_STRONG)

        # Панель
        self.panel.draw(seed=self.seed, track_ms=self.last_generation_ms)

        # FPS у верхньому правому куті viewport
        if self.frame_times:
            avg_frame = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame if avg_frame > 0 else 0.0
            self._fps_text.text = f"{fps:.0f} FPS"
            self._fps_text.draw()

        self._title_text.draw()

    def on_update(self, delta_time: float) -> None:
        now = time.perf_counter()
        self.frame_times.append(now - self._last_frame_stamp)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        self._last_frame_stamp = now

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self.panel.on_mouse_motion(x, y)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        action = self.panel.on_mouse_press(x, y)
        if action == "new_track":
            self._generate_new_track(seed=random.randint(0, 1_000_000))
        elif action == "same_seed" and self.seed is not None:
            self._generate_new_track(seed=self.seed)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == arcade.key.R:
            self._generate_new_track(seed=random.randint(0, 1_000_000))


def main() -> None:
    window = CarAIWindow()
    arcade.run()


if __name__ == "__main__":
    main()
