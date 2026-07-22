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
import arcade.gui
import numpy as np

import theme


def seed_from_text(text: str) -> int:
    """Перетворює будь-який текст (цифри, кирилиця, символи, довільна довжина)
    у детермінований цілочисельний seed через коди символів (ASCII/Unicode).
    Той самий текст завжди дає той самий seed. numpy.random.default_rng вимагає
    невід'ємний seed, тому від'ємні числа переводяться в додатні через abs()."""
    if text.strip() == "":
        raise ValueError("empty text")
    if text.strip().lstrip("-").isdigit():
        return abs(int(text.strip()))
    # Поліноміальний хеш по кодових точках символів (стабільний, на відміну від
    # вбудованого hash(), який рандомізується між запусками Python).
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_000_007
    return value


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

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._text.x = x
        self._text.y = y

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


class IconButton:
    """Маленька квадратна кнопка з намальованою (не текстовою/emoji) іконкою —
    copy (дві прямокутні рамки внахлест) або paste (планшет з галочкою)."""

    def __init__(self, x: float, y: float, size: float, icon: str):
        self.x = x
        self.y = y
        self.size = size
        self.icon = icon  # "copy" | "paste"
        self.hovered = False
        self.flash_timer = 0.0  # >0 — коротка підсвітка після успішної дії

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def contains(self, x: float, y: float) -> bool:
        h = self.size / 2
        return (self.x - h <= x <= self.x + h) and (self.y - h <= y <= self.y + h)

    def trigger_flash(self) -> None:
        self.flash_timer = 0.35

    def update(self, delta_time: float) -> None:
        if self.flash_timer > 0:
            self.flash_timer = max(0.0, self.flash_timer - delta_time)

    def draw(self) -> None:
        h = self.size / 2
        left, right = self.x - h, self.x + h
        bottom, top = self.y - h, self.y + h

        if self.flash_timer > 0:
            fill, border, ink = theme.ACCENT_SOFT, theme.ACCENT, theme.ACCENT_STRONG
        elif self.hovered:
            fill, border, ink = theme.BG_CARD, theme.ACCENT, theme.ACCENT_STRONG
        else:
            fill, border, ink = theme.BG_CARD, theme.BORDER, theme.TEXT_SECONDARY

        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, fill)
        arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, border, border_width=1.5)

        cx, cy = self.x, self.y
        if self.icon == "reset":
            # кругова стрілка проти годинникової — "скинути до дефолту"
            s = self.size * 0.28
            arcade.draw_arc_outline(cx, cy, s * 2, s * 2, ink, start_angle=-60, end_angle=200, border_width=1.8)
            tip_angle = np.radians(200)
            tx = cx + s * np.cos(tip_angle)
            ty = cy + s * np.sin(tip_angle)
            arcade.draw_triangle_filled(tx - 3, ty + 1, tx + 3, ty + 3, tx + 1, ty - 4, ink)
        elif self.icon == "copy":
            # дві рамки внахлест — класичний піктограма "копіювати"
            s = self.size * 0.22
            arcade.draw_lrbt_rectangle_outline(cx - s * 1.4, cx + s * 0.6, cy - s * 0.6, cy + s * 1.4, ink, border_width=1.4)
            arcade.draw_lrbt_rectangle_filled(cx - s * 0.6, cx + s * 1.4, cy - s * 1.4, cy + s * 0.6, theme.BG_CARD)
            arcade.draw_lrbt_rectangle_outline(cx - s * 0.6, cx + s * 1.4, cy - s * 1.4, cy + s * 0.6, ink, border_width=1.4)
        else:  # "paste"
            # планшет з галочкою — піктограма "вставити"
            s = self.size * 0.24
            arcade.draw_lrbt_rectangle_outline(cx - s, cx + s, cy - s * 1.3, cy + s * 1.1, ink, border_width=1.4)
            arcade.draw_line(cx - s * 0.5, cy + s * 1.1, cx + s * 0.5, cy + s * 1.1, ink, line_width=2.2)
            arcade.draw_line(cx - s * 0.5, cy - s * 0.15, cx - s * 0.1, cy - s * 0.55, ink, line_width=1.6)
            arcade.draw_line(cx - s * 0.1, cy - s * 0.55, cx + s * 0.6, cy + s * 0.35, ink, line_width=1.6)


class Checkbox:
    """Квадратик з галочкою (намальованою, не текстовою) + підпис праворуч."""

    def __init__(self, x: float, y: float, size: float, label: str, checked: bool = False):
        self.x = x
        self.y = y
        self.size = size
        self.checked = checked
        self.hovered = False
        self._text = arcade.Text(
            label, x + size / 2 + 8, y, theme.TEXT_SECONDARY, font_size=11.5,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._text.x = x + self.size / 2 + 8
        self._text.y = y

    def contains(self, x: float, y: float) -> bool:
        h = self.size / 2 + 6  # трохи більша ціль для кліку, зручніше влучити
        return (self.x - h <= x <= self.x + h) and (self.y - h <= y <= self.y + h)

    def draw(self) -> None:
        h = self.size / 2
        left, right = self.x - h, self.x + h
        bottom, top = self.y - h, self.y + h

        border = theme.ACCENT if self.hovered else theme.BORDER
        fill = theme.ACCENT_SOFT if self.checked else theme.BG_CARD
        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, fill)
        arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, border, border_width=1.5)

        if self.checked:
            arcade.draw_line(left + h * 0.3, self.y, self.x - h * 0.1, bottom + h * 0.35, theme.ACCENT_STRONG, line_width=2.0)
            arcade.draw_line(self.x - h * 0.1, bottom + h * 0.35, right - h * 0.15, top - h * 0.2, theme.ACCENT_STRONG, line_width=2.0)

        self._text.draw()


class Stepper:
    """Числове значення з кнопками −/+ по боках — для цілочисельних параметрів
    (кількість ботів), де повзунок був би незручним."""

    def __init__(self, x: float, y: float, width: float, value: int, min_value: int, max_value: int, step: int = 1):
        self.x = x
        self.y = y
        self.width = width
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.enabled = True
        self.btn_size = 24
        self._minus_hovered = False
        self._plus_hovered = False
        self._value_text = arcade.Text(
            str(value), x, y, theme.TEXT_PRIMARY, font_size=13,
            anchor_x="center", anchor_y="center", font_name=theme.FONT, bold=True,
        )
        self._minus_text = arcade.Text(
            "-", x, y, theme.TEXT_SECONDARY, font_size=15,
            anchor_x="center", anchor_y="center", font_name=theme.FONT,
        )
        self._plus_text = arcade.Text(
            "+", x, y, theme.TEXT_SECONDARY, font_size=15,
            anchor_x="center", anchor_y="center", font_name=theme.FONT,
        )

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._value_text.x = x
        self._value_text.y = y
        self._minus_text.x = x - self.width / 2 + self.btn_size / 2
        self._minus_text.y = y
        self._plus_text.x = x + self.width / 2 - self.btn_size / 2
        self._plus_text.y = y

    def _minus_bounds(self) -> tuple[float, float, float, float]:
        cx = self.x - self.width / 2 + self.btn_size / 2
        h = self.btn_size / 2
        return cx - h, cx + h, self.y - h, self.y + h

    def _plus_bounds(self) -> tuple[float, float, float, float]:
        cx = self.x + self.width / 2 - self.btn_size / 2
        h = self.btn_size / 2
        return cx - h, cx + h, self.y - h, self.y + h

    def draw(self) -> None:
        left, right = self.x - self.width / 2, self.x + self.width / 2
        bottom, top = self.y - self.btn_size / 2, self.y + self.btn_size / 2
        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, theme.BG_CARD)
        arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, theme.BORDER, border_width=1.5)

        text_color = theme.TEXT_PRIMARY if self.enabled else theme.TEXT_FAINT
        self._value_text.color = text_color
        self._value_text.text = str(self.value)
        self._value_text.draw()

        for bounds, text, hovered in (
            (self._minus_bounds(), self._minus_text, self._minus_hovered),
            (self._plus_bounds(), self._plus_text, self._plus_hovered),
        ):
            b_left, b_right, b_bottom, b_top = bounds
            fill = theme.ACCENT_SOFT if (hovered and self.enabled) else theme.BG_RAISED
            arcade.draw_lrbt_rectangle_filled(b_left, b_right, b_bottom, b_top, fill)
            text.color = theme.ACCENT_STRONG if (hovered and self.enabled) else text_color
            text.draw()

    def on_mouse_motion(self, x: float, y: float) -> None:
        ml, mr, mb, mt = self._minus_bounds()
        pl, pr, pb, pt = self._plus_bounds()
        self._minus_hovered = ml <= x <= mr and mb <= y <= mt
        self._plus_hovered = pl <= x <= pr and pb <= y <= pt

    def on_mouse_press(self, x: float, y: float) -> bool:
        """Повертає True, якщо клік влучив і значення могло змінитись."""
        if not self.enabled:
            return False
        ml, mr, mb, mt = self._minus_bounds()
        if ml <= x <= mr and mb <= y <= mt:
            self.value = max(self.min_value, self.value - self.step)
            return True
        pl, pr, pb, pt = self._plus_bounds()
        if pl <= x <= pr and pb <= y <= pt:
            self.value = min(self.max_value, self.value + self.step)
            return True
        return False


class LabeledSlider:
    """UISlider + текстовий підпис над ним (назва + поточне значення)."""

    def __init__(self, ui_manager: arcade.gui.UIManager, label: str,
                 value: float, min_value: float, max_value: float, step: float = 0.1):
        self.label = label
        self.slider = arcade.gui.UISlider(
            value=value, min_value=min_value, max_value=max_value, step=step,
            x=0, y=0, width=10, height=16,
        )
        ui_manager.add(self.slider)
        self._label_text = arcade.Text(
            "", 0, 0, theme.TEXT_SECONDARY, font_size=10.5,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )

    @property
    def value(self) -> float:
        return self.slider.value

    @property
    def enabled(self) -> bool:
        return not self.slider.disabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.slider.disabled = not value

    def move(self, x: float, width: float, y: float) -> None:
        self.slider.rect = self.slider.rect.align_left(x - width / 2).align_top(y + 8)
        self.slider.width = width
        self._label_text.x = x - width / 2
        self._label_text.y = y + 18

    def draw(self) -> None:
        self._label_text.color = theme.TEXT_SECONDARY if self.enabled else theme.TEXT_FAINT
        self._label_text.text = f"{self.label}: {self.value:.1f}x"
        self._label_text.draw()


class TabBar:
    """Верхній перемикач режимів вікна ("Навчання" / "Заїзди") — дві рівні
    вкладки, активна підсвічена акцентним кольором. Сам перемикач НЕ знає,
    що саме означають режими — лише повертає обраний ключ, як і Button."""

    def __init__(self, x: float, y: float, width: float, height: float, tabs: list[tuple[str, str]]):
        # tabs: [(key, label), ...]
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.tabs = tabs
        self.active_key = tabs[0][0]
        self._texts = [
            arcade.Text(
                label, 0, 0, theme.TEXT_SECONDARY, font_size=12.5,
                anchor_x="center", anchor_y="center", font_name=theme.FONT, bold=True,
            )
            for _, label in tabs
        ]
        self.move(x, y)

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        tab_w = self.width / len(self.tabs)
        for i, text in enumerate(self._texts):
            text.x = x - self.width / 2 + tab_w * (i + 0.5)
            text.y = y

    def _tab_bounds(self, index: int) -> tuple[float, float, float, float]:
        tab_w = self.width / len(self.tabs)
        left = self.x - self.width / 2 + tab_w * index
        return left, left + tab_w, self.y - self.height / 2, self.y + self.height / 2

    def draw(self) -> None:
        left = self.x - self.width / 2
        right = self.x + self.width / 2
        bottom = self.y - self.height / 2
        top = self.y + self.height / 2
        arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, theme.BG_CARD)
        arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, theme.BORDER, border_width=1.5)

        for i, (key, _) in enumerate(self.tabs):
            t_left, t_right, t_bottom, t_top = self._tab_bounds(i)
            is_active = key == self.active_key
            if is_active:
                arcade.draw_lrbt_rectangle_filled(t_left, t_right, t_bottom, t_top, theme.ACCENT_SOFT)
            self._texts[i].color = theme.ACCENT_STRONG if is_active else theme.TEXT_SECONDARY
            self._texts[i].draw()
            if i > 0:
                arcade.draw_line(t_left, bottom, t_left, top, theme.BORDER, line_width=1)

    def on_mouse_press(self, x: float, y: float) -> str | None:
        for i, (key, _) in enumerate(self.tabs):
            t_left, t_right, t_bottom, t_top = self._tab_bounds(i)
            if t_left <= x <= t_right and t_bottom <= y <= t_top:
                self.active_key = key
                return key
        return None


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

    def move(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self._text.x = x - self.width / 2
        self._text.y = y

    def draw(self) -> None:
        self._text.draw()
        arcade.draw_line(
            self.x - self.width / 2, self.y - 12, self.x + self.width / 2, self.y - 12,
            theme.BORDER_SOFT, line_width=1,
        )


class SidePanel:
    """Права панель — трек-контроли (активні) + заглушки майбутніх етапів."""

    def __init__(self, window: arcade.Window, panel_x: float, panel_width: float, window_height: float):
        self.window = window
        self.panel_width = panel_width
        self.ui_manager = arcade.gui.UIManager(window)
        self.ui_manager.enable()

        self.sections: list[Section] = []
        self.buttons: dict[str, Button] = {}

        self.seed_input = arcade.gui.UIInputText(
            x=0, y=0, width=10, height=30, text="", font_size=12,
            text_color=theme.TEXT_PRIMARY, caret_color=theme.ACCENT, border_color=theme.BORDER,
        ).with_background(color=arcade.types.Color(*theme.BG_CARD)).with_padding(left=8, top=6)
        self.ui_manager.add(self.seed_input)
        self.paste_icon = IconButton(0, 0, 30, "paste")

        self._seed_result_value = ""  # поточний seed як текст — джерело для копіювання
        self._seed_result_bg_rect: tuple[float, float, float, float] = (0, 0, 0, 0)  # left, right, bottom, top
        self._seed_result_text = arcade.Text(
            "", 0, 0, theme.TEXT_SECONDARY, font_size=12,
            anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )
        self.copy_icon = IconButton(0, 0, 26, "copy")

        self._seed_input_hint = arcade.Text(
            "seed (текст/число)…", 0, 0, theme.TEXT_FAINT,
            font_size=11, anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )
        self._seed_result_label = arcade.Text(
            "поточний seed (клік → Ctrl+C):", 0, 0, theme.TEXT_FAINT,
            font_size=10.5, anchor_x="left", font_name=theme.FONT,
        )
        self._gen_time_text = arcade.Text(
            "", 0, 0, theme.TEXT_FAINT, font_size=11, anchor_x="left", font_name=theme.FONT,
        )

        # Швидкість симуляції під час RL-тренування — фіксований список
        # значень (не повзунок) — клік по плашці перемикає на наступне.
        self.speed_options = [1, 3, 5, 10, 15, 20, 30, 50, 100]
        self.speed_option_idx = 0
        self._speed_label = arcade.Text(
            "швидкість тренування: 1x", 0, 0, theme.TEXT_PRIMARY, font_size=11,
            anchor_x="center", anchor_y="center", font_name=theme.FONT,
        )
        self._speed_hint_text = arcade.Text(
            "клікни, щоб змінити", 0, 0, theme.TEXT_FAINT, font_size=9,
            anchor_x="center", anchor_y="center", font_name=theme.FONT,
        )

        # Галочка "показувати промені бота" — окремо від запуску навчання,
        # можна вмикати/вимикати будь-коли, щоб візуально перевірити raycast.
        self.show_rays_checkbox = Checkbox(0, 0, 18, "показувати промені бота", checked=False)

        # --- Налаштування навчання (перед стартом) ---
        self._bot_count_label = arcade.Text(
            "ботів (паралельно)", 0, 0, theme.TEXT_SECONDARY, font_size=10.5,
            anchor_x="center", anchor_y="center", font_name=theme.FONT,
        )
        # max_value=24 — вище цього кожен додатковий бот це окремий OS-процес
        # з власною копією torch у пам'яті (Windows spawn, не fork); 50+
        # ботів на звичайній машині впирається в ліміт файлу підкачки і
        # валить процес MemoryError-ами (перевірено на практиці).
        # Слайдери ваг reward звідси прибрано — всі налаштування нагород
        # тепер у власній вкладці "Налаштування" (settings_ui.py).
        self.bot_count_stepper = Stepper(0, 0, width=140, value=16, min_value=1, max_value=24, step=1)

        # --- Збереження моделі: назва для запису + перемикач для вибору файлу
        # для завантаження (клік по "Завантажити" гортає список наявних) ---
        self.model_name_input = arcade.gui.UIInputText(
            x=0, y=0, width=10, height=28, text="", font_size=12,
            text_color=theme.TEXT_PRIMARY, caret_color=theme.ACCENT, border_color=theme.BORDER,
        ).with_background(color=arcade.types.Color(*theme.BG_CARD)).with_padding(left=8, top=5)
        self.ui_manager.add(self.model_name_input)
        self._model_name_hint = arcade.Text(
            "назва моделі…", 0, 0, theme.TEXT_FAINT,
            font_size=11, anchor_x="left", anchor_y="center", font_name=theme.FONT,
        )
        self._load_selection_text = arcade.Text(
            "немає збережених моделей", 0, 0, theme.TEXT_FAINT, font_size=10.5,
            anchor_x="center", anchor_y="center", font_name=theme.FONT,
        )

        self.relayout(panel_x=panel_x, window_height=window_height)

    def relayout(self, panel_x: float, window_height: float) -> None:
        """Перераховує позиції всіх елементів панелі під новий розмір вікна —
        викликається і з __init__, і з on_resize()."""
        self.panel_x = panel_x
        self.window_height = window_height

        pad = 20
        content_x = panel_x
        content_width = self.panel_width - pad * 2
        y = window_height - 40

        first_pass = not self.sections
        section_idx = 0
        button_defs: list[tuple[str, float, float, float, float, str, bool]] = []
        # (key, x, y, w, h, label, enabled) — заповнюється нижче по мірі проходу

        def next_section(label: str) -> None:
            nonlocal section_idx
            if first_pass:
                self.sections.append(Section(content_x, y, content_width, label))
            else:
                self.sections[section_idx].width = content_width
                self.sections[section_idx].move(content_x, y)
            section_idx += 1

        def place_button(key: str, x: float, w: float, h: float, label: str, enabled: bool) -> None:
            if first_pass:
                self.buttons[key] = Button(x, y, w, h, label, enabled=enabled)
            else:
                btn = self.buttons[key]
                btn.width = w
                btn.height = h
                btn.move(x, y)

        # --- Секція: ТРАСА (активна на Етапі 1) ---
        next_section("ТРАСА")
        y -= 34

        icon_size = 30
        icon_gap = 6
        input_height = 30
        input_width = content_width - icon_size - icon_gap
        input_left = content_x - content_width / 2
        self.seed_input.rect = self.seed_input.rect.align_left(input_left).align_top(y + input_height / 2)
        self.seed_input.width = input_width
        self.seed_input.height = input_height
        self._seed_input_hint.x = input_left + 8
        self._seed_input_hint.y = y
        self.paste_icon.size = icon_size
        self.paste_icon.move(content_x + content_width / 2 - icon_size / 2, y)
        y -= 42

        # Одна кнопка: якщо поле заповнене — генерує за ним, якщо порожнє — випадково.
        place_button("new_track", content_x, content_width, 34, "Нова траса", True)
        y -= 56

        # --- Секція: КЕРУВАННЯ (Етап 2) ---
        next_section("КЕРУВАННЯ")
        y -= 34
        place_button("toggle_driving", content_x, content_width, 34, "Їздити самому: УВІМК", True)
        y -= 56

        # --- Секція: НАЛАШТУВАННЯ (доступні лише поки навчання не запущено) ---
        next_section("НАЛАШТУВАННЯ")
        y -= 30
        self._bot_count_label.x = content_x
        self._bot_count_label.y = y
        self.bot_count_stepper.move(content_x, y - 20)
        y -= 60

        # --- Секція: НАВЧАННЯ (Етап 4) ---
        next_section("НАВЧАННЯ")
        y -= 34
        place_button("train_start", content_x, content_width, 34, "Почати навчання", True)
        y -= 42

        # Швидкість симуляції під час тренування — клікабельна плашка з
        # фіксованим списком значень (speed_options), клік гортає далі по
        # колу. Впливає лише на RL-тренування, не на гру гравця.
        # Кнопка 32px заввишки з центром у y: підпис і підказка — ДВА рядки
        # всередині неї, центровані як пара (раніше підпис був на y-8, а
        # підказка взагалі під нижнім краєм кнопки — виглядало зсунутим).
        place_button("speed_option", content_x, content_width, 32, "", True)
        self._speed_label.x = content_x
        self._speed_label.y = y + 5
        self._speed_hint_text.x = content_x
        self._speed_hint_text.y = y - 8
        y -= 44

        self.show_rays_checkbox.move(content_x - content_width / 2 + 9, y)
        y -= 40

        # --- Секція: ЗБЕРЕЖЕННЯ (checkpoint однієї моделі на диск) ---
        next_section("ЗБЕРЕЖЕННЯ")
        y -= 30

        name_input_height = 28
        name_input_left = content_x - content_width / 2
        self.model_name_input.rect = self.model_name_input.rect.align_left(name_input_left).align_top(y + name_input_height / 2)
        self.model_name_input.width = content_width
        self.model_name_input.height = name_input_height
        self._model_name_hint.x = name_input_left + 8
        self._model_name_hint.y = y
        y -= 40

        place_button("save_model", content_x - content_width / 4, content_width / 2 - 4, 30, "Зберегти", True)
        place_button("load_model", content_x + content_width / 4, content_width / 2 - 4, 30, "Завантажити", True)
        y -= 26

        # Стрілки гортають, яка модель показана (і буде завантажена наступним
        # кліком "Завантажити") — окремо від самого завантаження, щоб можна
        # було спершу переглянути назви кількох файлів.
        arrow_w = 22
        place_button("load_prev", content_x - content_width / 2 + arrow_w / 2, arrow_w, 22, "◁", True)
        place_button("load_next", content_x + content_width / 2 - arrow_w / 2, arrow_w, 22, "▷", True)
        self._load_selection_text.x = content_x
        self._load_selection_text.y = y
        y -= 30

        self.info_y_bottom = y
        info_x = content_x - (self.panel_width - pad * 2) / 2

        self._seed_result_label.x = info_x
        self._seed_result_label.y = self.info_y_bottom

        result_icon_size = 26
        result_height = 26
        result_left = content_x - content_width / 2
        result_y = self.info_y_bottom - 22
        self._seed_result_bg_rect = (
            result_left, content_x + content_width / 2,
            result_y - result_height / 2, result_y + result_height / 2,
        )
        self._seed_result_text.x = result_left + 8
        self._seed_result_text.y = result_y
        self.copy_icon.size = result_icon_size
        self.copy_icon.move(content_x + content_width / 2 - result_icon_size / 2, result_y)

        self._gen_time_text.x = info_x
        self._gen_time_text.y = result_y - result_height - 10

    def draw(self, seed: int | None = None, track_ms: float | None = None, training_locked: bool = False,
              load_selection_label: str | None = None) -> None:
        left = self.panel_x - self.panel_width / 2
        right = self.panel_x + self.panel_width / 2
        arcade.draw_lrbt_rectangle_filled(left, right, 0, self.window_height, theme.BG_RAISED)
        arcade.draw_line(left, 0, left, self.window_height, theme.BORDER_SOFT, line_width=1)

        for section in self.sections:
            section.draw()
        for button in self.buttons.values():
            button.draw()

        self._seed_result_label.draw()
        self.ui_manager.draw()
        if not self.seed_input.text:
            self._seed_input_hint.draw()
        if not self.model_name_input.text:
            self._model_name_hint.draw()

        # Налаштування (кількість ботів, ваги reward) заблоковані, поки триває
        # навчання — зміна bot_count під час роботи вимагала б перестворення
        # SubprocVecEnv, а ваги reward все одно вже "запечені" у процесах.
        self.bot_count_stepper.enabled = not training_locked
        self._bot_count_label.color = theme.TEXT_SECONDARY if not training_locked else theme.TEXT_FAINT
        self._bot_count_label.draw()
        self.bot_count_stepper.draw()

        self._speed_label.text = f"швидкість тренування: {self.speed_multiplier}x"
        self._speed_label.draw()
        self._speed_hint_text.draw()

        self.show_rays_checkbox.draw()

        self._load_selection_text.text = load_selection_label or "немає збережених моделей"
        self._load_selection_text.draw()

        bg_left, bg_right, bg_bottom, bg_top = self._seed_result_bg_rect
        arcade.draw_lrbt_rectangle_filled(bg_left, bg_right, bg_bottom, bg_top, theme.BG_RAISED)
        arcade.draw_lrbt_rectangle_outline(bg_left, bg_right, bg_bottom, bg_top, theme.BORDER_SOFT, border_width=1.5)
        self._seed_result_text.draw()

        self.paste_icon.draw()
        self.copy_icon.draw()

        if seed is not None and self._seed_result_value != str(seed):
            self._seed_result_value = str(seed)
            self._seed_result_text.text = self._seed_result_value
        if track_ms is not None:
            self._gen_time_text.text = f"генерація: {track_ms:.1f} мс"
            self._gen_time_text.draw()

    def on_update(self, delta_time: float) -> None:
        self.paste_icon.update(delta_time)
        self.copy_icon.update(delta_time)

    def on_mouse_motion(self, x: float, y: float) -> None:
        for button in self.buttons.values():
            button.hovered = button.enabled and button.contains(x, y)
        self.paste_icon.hovered = self.paste_icon.contains(x, y)
        self.copy_icon.hovered = self.copy_icon.contains(x, y)
        self.show_rays_checkbox.hovered = self.show_rays_checkbox.contains(x, y)
        self.bot_count_stepper.on_mouse_motion(x, y)

    def on_mouse_press(self, x: float, y: float) -> str | None:
        if self.copy_icon.contains(x, y):
            self.window.set_clipboard_text(self._seed_result_value)
            self.copy_icon.trigger_flash()
            return None
        if self.paste_icon.contains(x, y):
            try:
                clip_text = self.window.get_clipboard_text()
            except Exception:
                clip_text = ""
            if clip_text:
                self.seed_input.text = clip_text.strip()
                self.paste_icon.trigger_flash()
            return None
        if self.show_rays_checkbox.contains(x, y):
            self.show_rays_checkbox.checked = not self.show_rays_checkbox.checked
            return None
        if self.bot_count_stepper.on_mouse_press(x, y):
            return None
        if self.buttons["speed_option"].contains(x, y):
            self.speed_option_idx = (self.speed_option_idx + 1) % len(self.speed_options)
            return None

        for name, button in self.buttons.items():
            if button.enabled and button.contains(x, y):
                return name
        return None

    @property
    def speed_multiplier(self) -> int:
        return self.speed_options[self.speed_option_idx]

    def take_seed_input_text(self) -> str:
        """Повертає введений у поле seed текст і одразу очищає поле — після
        генерації користувач бачить фінальний числовий seed внизу панелі."""
        text = self.seed_input.text
        self.seed_input.text = ""
        return text

    @property
    def model_name_text(self) -> str:
        """Поле НЕ очищається після збереження (на відміну від seed) — той
        самий підпис лишається, щоб можна було зберегти оновлену версію
        поверх файлу з тою самою назвою без повторного набору тексту."""
        return self.model_name_input.text.strip()
