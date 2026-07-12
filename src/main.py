"""CarAI — Етап 1: генерація траси + каркас UI.

Запуск: venv/Scripts/python.exe src/main.py
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import arcade
import numpy as np

import theme
from bot_panel import VISIBLE_COUNT_OPTIONS, BotPanel, ControlsIndicator
from car import CAR_LENGTH, CAR_WIDTH, CarState, step_car
from lap_tracker import GhostPlayer, LapTracker
from leaderboard import Leaderboard
from rl_env import RAY_ANGLES_DEG
from track_generator import Track, generate_track
from trainer import BackgroundTrainer, RewardWeights
from ui_panel import SidePanel, seed_from_text

BOT_PANEL_GAP = 12

DEFAULT_WINDOW_WIDTH = 1440
DEFAULT_WINDOW_HEIGHT = 900
PANEL_WIDTH = 300

LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "track_generation"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path(__file__).resolve().parent.parent / "logs" / "models"

logger = logging.getLogger("carai.track")
logger.setLevel(logging.INFO)
_log_file = LOG_DIR / f"{time.strftime('%Y-%m-%d')}_run.log"
_handler = logging.FileHandler(_log_file, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
if not logger.handlers:
    logger.addHandler(_handler)


class CarAIWindow(arcade.Window):
    def __init__(self) -> None:
        super().__init__(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, "CarAI", resizable=True)
        arcade.set_background_color(theme.BG)

        # Розміри вікна тепер атрибути екземпляра (не глобальні константи) —
        # on_resize() оновлює їх і перебудовує панель/рендер-кеш під новий розмір.
        self.window_width = DEFAULT_WINDOW_WIDTH
        self.window_height = DEFAULT_WINDOW_HEIGHT
        self.viewport_width = DEFAULT_WINDOW_WIDTH - PANEL_WIDTH

        self.panel = SidePanel(
            window=self,
            panel_x=self.viewport_width + PANEL_WIDTH / 2,
            panel_width=PANEL_WIDTH,
            window_height=self.window_height,
        )

        self.track: Track | None = None
        self.seed: int | None = None
        self.last_generation_ms: float | None = None
        self._asphalt_shapes: arcade.shape_list.ShapeElementList | None = None
        self._screen_scale: float = 1.0
        self._screen_center_world: tuple[float, float] = (0.0, 0.0)
        self._checkpoint_screen_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []

        # --- Ручне керування (Етап 2) ---
        self.car: CarState | None = None
        self._keys_pressed: set[int] = set()
        self.driving_enabled = True
        self.lap_tracker: LapTracker | None = None
        self.ghost_player: GhostPlayer | None = None
        self.leaderboard = Leaderboard(x=16, y=self.window_height - 48)

        # --- RL-навчання (Етап 4) ---
        self.trainer: BackgroundTrainer | None = None
        self.bot_panel = BotPanel(x=16, y=self.window_height - 48 - Leaderboard.HEIGHT - BOT_PANEL_GAP)
        self.controls_indicator = ControlsIndicator(x=16, y=48)
        self._load_selection_idx = 0  # який файл у logs/models/seed_<seed>/ обрано для наступного "Завантажити"

        self._fps_text = arcade.Text(
            "", self.viewport_width - 16, self.window_height - 16, theme.TEXT_FAINT,
            font_size=12, anchor_x="right", anchor_y="top", font_name=theme.FONT,
        )
        self._title_text = arcade.Text(
            "CarAI", 24, self.window_height - 16, theme.TEXT_PRIMARY, font_size=16,
            anchor_x="left", anchor_y="top", font_name=theme.FONT, bold=True,
        )

        # FPS-лічильник — ковзне середнє за останні N кадрів (закладено з Етапу 1,
        # знадобиться повноцінно на Етапі 4 для моніторингу продуктивності навчання)
        self.frame_times: list[float] = []
        self._last_frame_stamp = time.perf_counter()

        self._generate_new_track(seed=random.randint(0, 1_000_000))

    def _set_train_button_label(self, label: str) -> None:
        self.panel.buttons["train_start"].label = label
        self.panel.buttons["train_start"]._text.text = label

    def _models_dir(self) -> Path:
        # Модель — це навчений мозок, не прив'язаний до конкретної траси
        # (seed), тому всі моделі лежать в одній спільній папці, ідентифікуються
        # лише назвою — можна навчати на одній трасі, завантажувати на іншій.
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return MODELS_DIR

    def _list_saved_models(self) -> list[Path]:
        return sorted(self._models_dir().glob("*.zip"), key=lambda p: p.name.lower())

    def _current_load_selection_label(self) -> str | None:
        models = self._list_saved_models()
        if not models:
            return None
        idx = self._load_selection_idx % len(models)
        return models[idx].stem

    def _save_model(self) -> None:
        if self.trainer is None:
            return
        name = self.panel.model_name_text or "модель"
        path = self._models_dir() / f"{name}.zip"
        if self.trainer.save(str(path)):
            logger.info("Модель збережена: %s", path)
            self._load_selection_idx = 0

    def _cycle_load_selection(self, direction: int) -> None:
        """Гортає список збережених моделей (стрілки ◁/▷) — саме лише не
        завантажує нічого, тільки змінює, яка назва показана під кнопками.
        Завантаження — окрема дія (кнопка "Завантажити")."""
        models = self._list_saved_models()
        if not models:
            return
        self._load_selection_idx = (self._load_selection_idx + direction) % len(models)

    def _load_model(self) -> None:
        if self.trainer is None or self.trainer.is_running:
            return
        models = self._list_saved_models()
        if not models:
            logger.info("Немає збережених моделей")
            return
        path = models[self._load_selection_idx % len(models)]

        # Завантаження (як і старт) вимагає піднятого VecEnv — той самий
        # прийом, що й для start(): request_load() лише позначає шлях,
        # фактичний PPO.load() відбувається лінькво у фоновому потоці. Старий
        # trainer більше не використовуватиметься — shutdown() закриває його
        # SubprocVecEnv остаточно (на відміну від stop(), який лише призупиняє).
        self.trainer.shutdown()
        self.trainer = BackgroundTrainer(
            self.track, reference_time=self.lap_tracker.best_time if self.lap_tracker else None,
            bot_count=self.panel.bot_count_stepper.value,
            reward_weights=self._read_reward_weights(),
        )
        self.trainer.request_load(str(path))
        self.trainer.start()
        self._set_train_button_label("Запускається…")
        logger.info("Завантажено модель: %s", path)

    def _read_reward_weights(self) -> RewardWeights:
        return RewardWeights(
            checkpoint=self.panel.reward_checkpoint_slider.value,
            speed=self.panel.reward_speed_slider.value,
            out_of_bounds_penalty=self.panel.reward_penalty_slider.value,
        )

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

        # Нова траса — нові checkpoints, тому старий ghost/прогрес кола втрачають сенс.
        # Якщо для цього SEED вже є збережений найкращий заїзд з минулого запуску
        # гри — підвантажуємо його одразу, разом з ghost.
        self.lap_tracker = LapTracker(checkpoints=track.checkpoints.copy())
        if self.lap_tracker.load_best_from_disk(seed):
            self.ghost_player = GhostPlayer(frames=self.lap_tracker.best_ghost)
            logger.info("Завантажено збережений заїзд для seed=%s: %.2fс", seed, self.lap_tracker.best_time)
        else:
            self.ghost_player = None

        # Машина спавниться на гарантованій прямій ділянці (start_pos/start_dir),
        # дивлячись вздовж напрямку траси. Якщо "їздити самому" вимкнено — машини
        # немає взагалі (гравець явно вимкнув свою участь).
        if self.driving_enabled:
            heading = float(np.arctan2(track.start_dir[1], track.start_dir[0]))
            self.car = CarState(position=track.start_pos.copy(), heading=heading)
        else:
            self.car = None

        # Нова траса — геометрія змінилась, ботам треба показати нову. Якщо
        # trainer/VecEnv уже піднятий (навчання коли-небудь стартувало на цій
        # трасі, байдуже — активне зараз чи щойно зупинене) — оновлюємо лише
        # геометрію в ІСНУЮЧИХ процесах (update_track, дешево), той самий
        # навчений мозок продовжує на новій трасі. Перестворюємо
        # BackgroundTrainer лише коли VecEnv ще НІКОЛИ не піднімався (немає
        # чого зберігати) — тоді підхоплюємо актуальні bot_count/reward_weights
        # з панелі налаштувань.
        reference_time = self.lap_tracker.best_time if self.lap_tracker is not None else None
        if self.trainer is not None and self.trainer.vec_env is not None:
            self.trainer.update_track(track, reference_time=reference_time)
        else:
            if self.trainer is not None:
                self.trainer.shutdown()
            self.trainer = BackgroundTrainer(
                track, reference_time=reference_time,
                bot_count=self.panel.bot_count_stepper.value,
                reward_weights=self._read_reward_weights(),
            )
            self.panel.buttons["train_start"].label = "Почати навчання"
            self.panel.buttons["train_start"]._text.text = "Почати навчання"
        self.bot_panel.scroll_offset = 0.0
        self.bot_panel.selected_bot_id = None
        self._load_selection_idx = 0

        # Уся геометрія рендеру рахується ОДИН РАЗ тут, а не щокадру в on_draw —
        # після Чайкіна+Catmull-Rom center_line має тисячі точок, і сотні окремих
        # draw_polygon_filled() викликів щокадру валили FPS до ~5. ShapeElementList
        # батчить усі трикутники асфальту в один GPU draw call.
        self._rebuild_screen_transform()
        self._build_render_cache()

        logger.info(
            "seed=%s  точок=%d  час=%.1fмс  checkpoints=%d",
            seed, len(track.control_points), elapsed_ms, len(track.checkpoints),
        )

    def _build_render_cache(self) -> None:
        assert self.track is not None
        left = self._track_to_screen(self.track.left_edge)
        right = self._track_to_screen(self.track.right_edge)

        shapes = arcade.shape_list.ShapeElementList()
        n = len(left)
        for i in range(n):
            j = (i + 1) % n
            quad = [left[i], left[j], right[j], right[i]]
            shapes.append(arcade.shape_list.create_polygon(quad, theme.ASPHALT + (255,)))
        shapes.append(arcade.shape_list.create_line_strip(left + [left[0]], theme.ASPHALT_EDGE + (255,), line_width=2))
        shapes.append(arcade.shape_list.create_line_strip(right + [right[0]], theme.ASPHALT_EDGE + (255,), line_width=2))

        # Checkpoint-лінії НЕ кладемо в статичний кеш — колір наступного обов'язкового
        # checkpoint змінюється щокадру (зелена підсвітка), тому вони малюються окремо
        # в _draw_checkpoints(). Тут лише зберігаємо готові екранні координати країв
        # кожного checkpoint, щоб не перераховувати їх щоразу.
        self._checkpoint_screen_edges = [
            (left[int(idx)], right[int(idx)]) for idx in self.track.checkpoint_indices
        ]

        # Стартова лінія — шаховий (checkered) візерунок упоперек траси, у світових
        # координатах: кілька рядів чорно-білих клітинок, орієнтованих уздовж
        # start_dir (вздовж траси) і перпендикуляра до нього (впоперек ширини).
        start_dir = self.track.start_dir
        normal = np.array([-start_dir[1], start_dir[0]])
        half_width = float(np.linalg.norm(self.track.left_edge[0] - self.track.right_edge[0])) / 2.0
        cell_along = 6.0  # довжина клітинки вздовж траси (світові одиниці)
        cell_rows = 5      # кількість клітинок впоперек ширини
        cell_across = (half_width * 2.0) / cell_rows

        checker_world_points = []  # [(p1, p2, p3, p4, is_white), ...] по 4 кути кожної клітинки
        for row in range(cell_rows):
            offset_across = -half_width + row * cell_across
            base = self.track.start_pos + normal * offset_across
            p1 = base - start_dir * (cell_along / 2)
            p2 = base + start_dir * (cell_along / 2)
            p3 = p2 + normal * cell_across
            p4 = p1 + normal * cell_across
            checker_world_points.append((p1, p2, p3, p4, row % 2 == 0))

        for p1, p2, p3, p4, is_white in checker_world_points:
            quad_screen = self._track_to_screen([p1, p2, p3, p4])
            if len(quad_screen) < 4:
                continue
            color = (255, 255, 255, 255) if is_white else (20, 20, 20, 255)
            shapes.append(arcade.shape_list.create_polygon(quad_screen, color))

        self._asphalt_shapes = shapes

    def _rebuild_screen_transform(self) -> None:
        """Параметри перетворення світових координат в екранні рахуються ОДИН РАЗ
        (а не щокадру) — викликається при новій трасі й при resize вікна. Машина
        рухається щокадру і теж використовує ці кешовані параметри, тому не сканує
        весь left_edge/right_edge (тисячі точок) на кожному кадрі."""
        assert self.track is not None
        margin = 60
        avail_w = self.viewport_width - margin * 2
        avail_h = self.window_height - margin * 2

        xs = self.track.left_edge[:, 0].tolist() + self.track.right_edge[:, 0].tolist()
        ys = self.track.left_edge[:, 1].tolist() + self.track.right_edge[:, 1].tolist()
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale = min(avail_w / span_x, avail_h / span_y)

        self._screen_scale = scale
        self._screen_center_world = ((min_x + max_x) / 2, (min_y + max_y) / 2)

    def _world_to_screen_point(self, p) -> tuple[float, float]:
        cx, cy = self._screen_center_world
        offset_x = self.viewport_width / 2
        offset_y = self.window_height / 2
        return (offset_x + (p[0] - cx) * self._screen_scale, offset_y + (p[1] - cy) * self._screen_scale)

    def _track_to_screen(self, points):
        """Трек генерується у світових координатах з центром (0,0) — переносимо
        у видиму область viewport з відступами і масштабом (кешованими)."""
        if self.track is None or len(points) == 0:
            return []
        return [self._world_to_screen_point(p) for p in points]

    def _draw_checkpoints(self) -> None:
        if self.lap_tracker is None:
            return
        next_idx = self.lap_tracker.next_checkpoint_idx
        for idx, (p_left, p_right) in enumerate(self._checkpoint_screen_edges):
            color = theme.GOOD if idx == next_idx else theme.CYAN
            alpha = 220 if idx == next_idx else 130
            width = 3.0 if idx == next_idx else 1.5
            arcade.draw_line(p_left[0], p_left[1], p_right[0], p_right[1], color + (alpha,), line_width=width)

    def _draw_problematic_segments(self) -> None:
        """Червоне півколо довкола checkpoint, де боти стабільно вилітають
        (SegmentDifficultyTracker.problematic_segments()) — прозорість і
        товщина ростуть з death_rate, тьмяне ледь помітне на порозі 30%,
        насичене на майже 100% смертей. Кожен checkpoint — окрема природна
        точка з фіксованим інтервалом (checkpoint_count при генерації), тому
        сусідні поля самі по собі не накладаються одне на одне."""
        if self.trainer is None or self.track is None:
            return
        problematic = self.trainer.get_problematic_segments()
        if not problematic:
            return

        n = len(self.track.checkpoints)
        for segment_idx, death_rate in problematic.items():
            checkpoint = self.track.checkpoints[segment_idx]
            prev_checkpoint = self.track.checkpoints[(segment_idx - 1) % n]
            screen_center = self._world_to_screen_point(checkpoint)

            approach = checkpoint - prev_checkpoint
            facing_deg = float(np.degrees(np.arctan2(approach[1], approach[0])))

            radius_world = 26.0
            radius_screen = radius_world * self._screen_scale
            alpha = int(90 + 140 * min(death_rate, 1.0))
            width = 2.0 + 3.0 * min(death_rate, 1.0)
            arcade.draw_arc_outline(
                screen_center[0], screen_center[1],
                radius_screen * 2, radius_screen * 2,
                theme.BAD + (alpha,),
                start_angle=facing_deg - 90, end_angle=facing_deg + 90,
                border_width=width,
            )

    def _draw_ghost(self) -> None:
        if self.ghost_player is None or self.lap_tracker is None or not self.lap_tracker.lap_running:
            return
        frame = self.ghost_player.sample(self.lap_tracker.lap_time)
        if frame is None:
            return
        car_shape = np.array([[7.0, 3.5], [7.0, -3.5], [-7.0, -3.5], [-7.0, 3.5]])
        c, s = np.cos(frame.heading), np.sin(frame.heading)
        rot = np.array([[c, -s], [s, c]])
        corners = car_shape @ rot.T + np.array(frame.position)
        screen_corners = [self._world_to_screen_point(p) for p in corners]
        arcade.draw_polygon_filled(screen_corners, theme.TEXT_FAINT + (110,))
        arcade.draw_polygon_outline(screen_corners, theme.TEXT_SECONDARY + (180,), line_width=1.5)

    def _draw_car(self) -> None:
        assert self.car is not None
        corners = self.car.corners()
        screen_corners = [self._world_to_screen_point(p) for p in corners]
        arcade.draw_polygon_filled(screen_corners, theme.ACCENT_STRONG + (255,))
        arcade.draw_polygon_outline(screen_corners, theme.BG, line_width=1.5)

        # Трикутник-ніс — показує напрямок капота (heading), окремо від напрямку
        # фактичного руху (velocity), щоб під час заносу було видно різницю.
        nose_world = self.car.position + np.array(
            [np.cos(self.car.heading), np.sin(self.car.heading)]
        ) * 10.0
        nose_screen = self._world_to_screen_point(nose_world)
        center_screen = self._world_to_screen_point(self.car.position)
        arcade.draw_line(center_screen[0], center_screen[1], nose_screen[0], nose_screen[1], theme.BG, line_width=2)

    def _draw_bots(self) -> None:
        """Малює всіх ботів, чия кількість наразі обрана у BotPanel (список
        видимих обрізається за поточним reward — найкращі зверху, як і в
        самій панелі балів). Промені рендеряться лише за галочкою."""
        if self.trainer is None:
            return
        snapshots = self.trainer.get_snapshots()
        if not snapshots:
            return

        visible_count = VISIBLE_COUNT_OPTIONS[self.bot_panel.visible_count_idx]
        ranked = sorted(snapshots, key=lambda s: s.episode_reward, reverse=True)[:visible_count]

        show_rays = self.panel.show_rays_checkbox.checked
        selected_id = self.bot_panel.selected_bot_id
        car_shape = np.array([[CAR_LENGTH / 2, CAR_WIDTH / 2], [CAR_LENGTH / 2, -CAR_WIDTH / 2],
                               [-CAR_LENGTH / 2, -CAR_WIDTH / 2], [-CAR_LENGTH / 2, CAR_WIDTH / 2]])

        # Виділений бот малюється останнім (поверх решти), трохи більшим і
        # яскравішим кольором — щоб не губився серед інших машинок на трасі.
        for snapshot in sorted(ranked, key=lambda s: s.bot_id == selected_id):
            is_selected = snapshot.bot_id == selected_id
            if show_rays or is_selected:
                ray_alpha = 200 if is_selected else 90
                for dist, angle_deg in zip(snapshot.ray_distances, RAY_ANGLES_DEG):
                    angle = snapshot.heading + np.radians(angle_deg)
                    end_world = snapshot.position + np.array([np.cos(angle), np.sin(angle)]) * dist
                    start_screen = self._world_to_screen_point(snapshot.position)
                    end_screen = self._world_to_screen_point(end_world)
                    arcade.draw_line(start_screen[0], start_screen[1], end_screen[0], end_screen[1], theme.CYAN + (ray_alpha,), line_width=1.0)

            # Виділений бот НЕ малюється більшим за реальний хітбокс — раніше
            # тут був scale=1.5, і користувач помітив, що це виглядало так,
            # ніби виділення бота змінює його фізичні габарити (насправді ні:
            # car.py::corners() завжди рахує track limits від справжнього
            # розміру машини, рендер тут ніяк на це не впливає). Але щоб не
            # створювати навіть візуальної плутанини — виділяємо лише
            # кольором і товщою обвідкою, розмір завжди справжній.
            c, s = np.cos(snapshot.heading), np.sin(snapshot.heading)
            rot = np.array([[c, -s], [s, c]])
            corners = car_shape @ rot.T + snapshot.position
            screen_corners = [self._world_to_screen_point(p) for p in corners]
            fill_color = theme.ACCENT_STRONG if is_selected else theme.CYAN
            outline_width = 2.5 if is_selected else 1.2
            arcade.draw_polygon_filled(screen_corners, fill_color + (255,))
            arcade.draw_polygon_outline(screen_corners, theme.BG, line_width=outline_width)

    def on_draw(self) -> None:
        self.clear()

        # Viewport — трава
        arcade.draw_lrbt_rectangle_filled(0, self.viewport_width, 0, self.window_height, theme.GRASS)

        if self.track is not None and self._asphalt_shapes is not None:
            self._asphalt_shapes.draw()

        self._draw_checkpoints()
        self._draw_problematic_segments()
        self._draw_ghost()
        self._draw_bots()

        if self.car is not None:
            self._draw_car()

        if self.lap_tracker is not None:
            current_lap = self.lap_tracker.lap_time if self.lap_tracker.lap_running else None
            bot_best_time = self.trainer.best_lap_time if self.trainer is not None else None
            self.leaderboard.draw(
                best_time=self.lap_tracker.best_time, current_lap_time=current_lap,
                bot_best_time=bot_best_time,
            )

        snapshots = self.trainer.get_snapshots() if self.trainer is not None else []
        bots_for_panel = [
            (s.bot_id, s.name, s.episode_reward, s.best_lap_time, s.current_lap_time, s.lap_number)
            for s in snapshots
        ]
        total_timesteps = self.trainer.total_timesteps if self.trainer is not None else 0
        self.bot_panel.draw(bots_for_panel, total_timesteps=total_timesteps)

        selected = next((s for s in snapshots if s.bot_id == self.bot_panel.selected_bot_id), None)
        self.controls_indicator.draw(
            name=selected.name if selected else None,
            throttle=selected.throttle if selected else 0,
            steer=selected.steer if selected else 0.0,
            brake=selected.brake if selected else 0.0,
        )

        # Панель — bot_count/reward заблоковані не лише поки навчання
        # активно працює, а й після зупинки, якщо VecEnv вже піднятий: клік
        # "Почати навчання" тоді ПРОДОВЖУЄ той самий trainer (щоб не втрачати
        # навчений прогрес), а не перестворює його з новими параметрами.
        training_locked = self.trainer is not None and (self.trainer.is_running or self.trainer.vec_env is not None)
        self.panel.draw(
            seed=self.seed, track_ms=self.last_generation_ms, training_locked=training_locked,
            load_selection_label=self._current_load_selection_label(),
        )

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
        self.panel.on_update(delta_time)

        if self.trainer is not None:
            self.trainer.speed_multiplier = self.panel.speed_multiplier
            if self.trainer.is_running and self.panel.buttons["train_start"].label != "Зупинити навчання":
                self._set_train_button_label("Зупинити навчання")

            # Боти сумарно назбирали AUTO_NEW_TRACK_AFTER_CYCLES успішних
            # циклів (MAX_LAPS_PER_EPISODE кіл поспіль без вильоту) — той
            # самий шлях, що й ручний клік "Нова траса": той самий навчений
            # мозок продовжує на новій трасі, лічильник скидається.
            if self.trainer.auto_new_track_requested:
                logger.info("Автоматична зміна траси: %d успішних циклів", self.trainer.cycles_completed)
                self.trainer.acknowledge_auto_new_track()
                self._generate_new_track(seed=random.randint(0, 1_000_000))

        if self.car is not None:
            throttle = 1 if arcade.key.W in self._keys_pressed else 0
            steer = 0.0
            if arcade.key.A in self._keys_pressed:
                steer += 1.0
            if arcade.key.D in self._keys_pressed:
                steer -= 1.0
            brake = 1.0 if arcade.key.SPACE in self._keys_pressed else 0.0
            # dt обмежується зверху — при короткому зависанні (напр. resize вікна)
            # величезний delta_time не повинен "телепортувати" машину крізь трасу.
            dt = min(delta_time, 1 / 20)
            self.car = step_car(self.car, throttle=throttle, steer=steer, brake=brake, dt=dt)

            if self.lap_tracker is not None:
                had_best_before = self.lap_tracker.best_ghost
                lap_completed = self.lap_tracker.update(
                    self.car.position, self.car.heading, dt, speed=self.car.speed
                )
                if lap_completed and self.lap_tracker.best_ghost is not had_best_before:
                    # Найкращий заїзд щойно оновився — привид з наступного кола
                    # відтворює саме цей запис, і результат одразу йде на диск
                    # (прив'язаний до seed — переживе перезапуск гри).
                    self.ghost_player = GhostPlayer(frames=self.lap_tracker.best_ghost)
                    if self.seed is not None:
                        self.lap_tracker.save_best_to_disk(self.seed)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.window_width = width
        self.window_height = height
        self.viewport_width = width - PANEL_WIDTH

        self.panel.relayout(
            panel_x=self.viewport_width + PANEL_WIDTH / 2,
            window_height=self.window_height,
        )
        self._fps_text.x = self.viewport_width - 16
        self._fps_text.y = self.window_height - 16
        self._title_text.y = self.window_height - 16
        self.leaderboard.move(x=16, y=self.window_height - 48)
        self.bot_panel.move(x=16, y=self.window_height - 48 - Leaderboard.HEIGHT - BOT_PANEL_GAP)

        if self.track is not None:
            self._rebuild_screen_transform()
            self._build_render_cache()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self.panel.on_mouse_motion(x, y)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.bot_panel.on_mouse_scroll(x, y, scroll_y)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        self.bot_panel.on_mouse_press(x, y)
        action = self.panel.on_mouse_press(x, y)
        if action == "new_track":
            text = self.panel.take_seed_input_text()
            if text.strip() == "":
                self._generate_new_track(seed=random.randint(0, 1_000_000))
            else:
                try:
                    seed = seed_from_text(text)
                except ValueError:
                    seed = random.randint(0, 1_000_000)
                self._generate_new_track(seed=seed)
        elif action == "toggle_driving":
            self.driving_enabled = not self.driving_enabled
            label = "Їздити самому: УВІМК" if self.driving_enabled else "Їздити самому: ВИМК"
            self.panel.buttons["toggle_driving"].label = label
            self.panel.buttons["toggle_driving"]._text.text = label

            if self.driving_enabled and self.track is not None:
                heading = float(np.arctan2(self.track.start_dir[1], self.track.start_dir[0]))
                self.car = CarState(position=self.track.start_pos.copy(), heading=heading)
                if self.lap_tracker is not None:
                    self.lap_tracker.reset_progress()
            elif not self.driving_enabled:
                self.car = None
                self._keys_pressed.clear()
        elif action == "train_start":
            if self.trainer is not None and not self.trainer.is_starting:
                if self.trainer.is_running:
                    self.trainer.stop()
                    self._set_train_button_label("Почати навчання")
                elif self.trainer.vec_env is not None:
                    # Навчання вже колись стартувало на цій трасі (модель з
                    # усіма навченими вагами й досі жива в self.trainer) —
                    # просто продовжуємо ТОЙ САМИЙ trainer, а не створюємо
                    # нового з нуля. Раніше "Зупинити" → "Почати" завжди
                    # перестворював BackgroundTrainer, тому прогрес губився
                    # щоразу — виправлено за проханням користувача.
                    self.trainer.start()
                    self._set_train_button_label("Запускається…")
                else:
                    # Навчання на цій трасі не запускалось ЖОДНОГО разу —
                    # немає що продовжувати, створюємо новий trainer з
                    # актуальними налаштуваннями панелі (bot_count/reward).
                    self.trainer = BackgroundTrainer(
                        self.track, reference_time=self.trainer.reference_time,
                        bot_count=self.panel.bot_count_stepper.value,
                        reward_weights=self._read_reward_weights(),
                    )
                    self.trainer.start()
                    # Піднімання N паралельних процесів на Windows займає до
                    # ~20с (немає fork, кожен процес — холодний старт нового
                    # інтерпретера) — без цього напису кнопка виглядала б
                    # "мертвою" весь цей час. on_update() підмінить текст на
                    # "Зупинити навчання", щойно is_running стане True.
                    self._set_train_button_label("Запускається…")
        elif action == "save_model":
            self._save_model()
        elif action == "load_model":
            self._load_model()
        elif action == "load_prev":
            self._cycle_load_selection(-1)
        elif action == "load_next":
            self._cycle_load_selection(1)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        # Поки якесь текстове поле активне (seed чи назва моделі), WASD має
        # друкувати текст, а не керувати машиною.
        if self.panel.seed_input.active or self.panel.model_name_input.active:
            return
        if symbol == arcade.key.R:
            self._generate_new_track(seed=random.randint(0, 1_000_000))
        self._keys_pressed.add(symbol)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._keys_pressed.discard(symbol)

    def on_close(self) -> None:
        if self.trainer is not None:
            self.trainer.shutdown()  # вікно закривається назавжди — реально прибрати OS-процеси
        super().on_close()


def main() -> None:
    window = CarAIWindow()
    arcade.run()


if __name__ == "__main__":
    main()
