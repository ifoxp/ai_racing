"""Gymnasium-середовище для RL-навчання. Обгортає фізику машини (car.py) і
трасу (track_generator.py) у стандартний RL-інтерфейс (reset/step).

Observation: raycast-датчики (промені від машини до межі асфальту в кількох
напрямках) + швидкість + напрямок і відстань до наступного checkpoint.
Компактний вектор чисел — швидко вчиться, не потребує GPU для самого
спостереження (на відміну від піксельного підходу).

Action: MultiDiscrete — газ (0/1), кермо (N рівнів -1..1), гальмо (M рівнів 0..1).

Reward: dense potential (наближення до checkpoint) + checkpoint bonus +
швидкість + track limits (4 кути хітбоксу, пільговий період) + фінальний
бонус за перемогу над часом гравця — узгоджена схема з користувачем.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from car import CAR_LENGTH, CAR_WIDTH, CarState, step_car
from track_generator import Track

# --- Raycast ---
RAY_MAX_DISTANCE = 200.0
# Передньо-бокове віяло (відносно heading, напрямку капота) — нерівномірне:
# густіше по центру (±5°), рідше до країв (±35°, ±50°, ±90°). Променя точно
# на 0° немає свідомо (запит користувача): ±5° і так майже дивляться вперед,
# а третій промінь у тому ж пучку (0°) дублював би їх — сильно корельований
# вхід, який лише витрачає ємність мережі, не додаючи інформації.
_FRONT_SIDE_ANGLES_DEG = [-90, -50, -35, -5, 5, 35, 50, 90]
# Задні промені (±135°, 180°) — не для звичайної їзди (перед машиною завжди
# важливіше), а для дрифту: коли velocity відхиляється від heading, саме
# задня частина корпусу заносить до бордюру з боку, куди машину несе. Без
# них бот не бачив, що "жопа" вже майже за межею траси під час заносу.
_REAR_ANGLES_DEG = [-180, -135, 135]
RAY_ANGLES_DEG = np.array(_FRONT_SIDE_ANGLES_DEG + _REAR_ANGLES_DEG, dtype=np.float64)
RAY_COUNT = len(RAY_ANGLES_DEG)

# --- Action discretization ---
STEER_LEVELS = 11   # -100%..100% з кроком 20% (-1.0, -0.8, ..., 0.8, 1.0)
BRAKE_LEVELS = 2     # бінарне — 0% або 100%, як і газ. Простір дій 2×11×2=44
                      # замість колишніх 2×11×6=132 — менше комбінацій, бот
                      # швидше натрапляє випадковим чином на робочу дію на
                      # поворотах (користувач: 5хв не вистачило на поворот 90°)

# --- Reward ---
CHECKPOINT_BONUS = 1.0
SPEED_REWARD_SCALE = 0.002
OUT_OF_BOUNDS_PENALTY = -10.0
MAX_EPISODE_SECONDS = 60.0

# Коло само по собі більше НЕ завершує епізод — бот продовжує їхати далі
# (не втрачає накопичений reward, стимул "берегти й нарощувати очки" замість
# "закінчити і почати спочатку"). Епізод завершується або при виїзді за межі,
# або після MAX_LAPS_PER_EPISODE кіл поспіль без вильоту, або по таймауту
# MAX_EPISODE_SECONDS — інакше rollout buffer PPO ніколи не бачив би межі
# епізоду і алгоритм втратив би сенс "де закінчується задача".
MAX_LAPS_PER_EPISODE = 3

# Штраф за тривалу їзду значно повільніше за власний нещодавній максимум —
# не карає уповільнення в повороті (SPEED_WINDOW_SECONDS ковзний максимум
# ловить "я щойно міг їхати швидше"), але карає застрягання/повзання.
SPEED_WINDOW_SECONDS = 5.0
SPEED_LOW_FRACTION = 0.5     # поріг: "повільно" = менше половини недавнього максимуму
SPEED_LOW_GRACE_SECONDS = 2.0  # стільки можна їхати повільно без кари (розгін після старту/повороту)
SPEED_LOW_PENALTY = -0.05     # щокадровий штраф, поки триває

# Прямий бонус за гальмування перед гострим поворотом — без нього PPO
# знаходить "простіший" спосіб пройти поворот (просто відпустити газ, drag
# сам сповільнює) і ніколи випадково не натрапляє на явно кращу комбінацію
# газ+гальмо+кермо, щоб цю звичку закріпити. Кут повороту рахується між
# напрямком (машина→checkpoint) і (checkpoint→наступний checkpoint) —
# наскільки різко траса зламається одразу ПІСЛЯ поточної цілі.
#
# Бонус пропорційний ПОТОЧНІЙ швидкості (не фіксований!) — на низькій
# швидкості гальмувати нема сенсу (dense reward від сповільнення й так
# переважує будь-який фіксований бонус), а на високій це саме те, де
# гальмування критичне для проходження повороту без вильоту.
SHARP_TURN_ANGLE_DEG = 45.0     # від цього кута поворот вважається "гострим"
SHARP_TURN_BRAKE_DISTANCE = 80.0  # у межах цієї відстані до checkpoint бонус активний
SHARP_TURN_BRAKE_SCALE = 0.01    # бонус = SHARP_TURN_BRAKE_SCALE * швидкість, поки brake=1 в зоні


def _build_edge_segments(left_edge: np.ndarray, right_edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Готує всі відрізки межі траси (обидва краї, замкнені в цикл) як масиви
    p1/p2 для векторизованого перетину з променями. Викликається ОДИН РАЗ при
    завантаженні траси, не щокадру — cast_rays() лише перевикористовує результат."""
    def segments(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p1 = edge
        p2 = np.roll(edge, -1, axis=0)
        return p1, p2

    l1, l2 = segments(left_edge)
    r1, r2 = segments(right_edge)
    return np.concatenate([l1, r1], axis=0), np.concatenate([l2, r2], axis=0)


def cast_rays_vectorized(position: np.ndarray, heading: float,
                          seg_p1: np.ndarray, seg_p2: np.ndarray,
                          max_distance: float = RAY_MAX_DISTANCE,
                          local_radius: float = RAY_MAX_DISTANCE * 1.5) -> np.ndarray:
    """Векторизована версія: усі промені проти всіх (локально відібраних)
    відрізків одним numpy-проходом, без Python-циклів. ~100x швидше за наївний
    подвійний цикл (критично — викликається щокадру для кожного агента)."""
    # Локальний відбір: тільки відрізки, чий будь-який кінець близько до машини.
    mid = (seg_p1 + seg_p2) / 2.0
    nearby = np.linalg.norm(mid - position, axis=1) <= local_radius
    p1 = seg_p1[nearby]
    p2 = seg_p2[nearby]
    if len(p1) == 0:
        return np.full(RAY_COUNT, max_distance)

    angles = heading + np.radians(RAY_ANGLES_DEG)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])  # (RAY_COUNT, 2)

    v2 = p2 - p1  # (S, 2) — напрямки відрізків
    v1 = position[None, :] - p1  # (S, 2)

    # v3 = перпендикуляр до напрямку променя, для кожного (ray, segment) окремо.
    # denom[r, s] = dot(v2[s], perp(direction[r]))
    perp_dirs = np.column_stack([-directions[:, 1], directions[:, 0]])  # (R, 2)
    denom = perp_dirs @ v2.T  # (R, S)

    valid = np.abs(denom) > 1e-9
    safe_denom = np.where(valid, denom, 1.0)

    # t1[r, s] = (v2 x v1) / denom  — відстань уздовж променя
    cross_v2_v1 = v2[:, 0][None, :] * v1[:, 1][None, :] - v2[:, 1][None, :] * v1[:, 0][None, :]
    t1 = cross_v2_v1 / safe_denom  # (R, S)

    # t2[r, s] = dot(v1[s], perp(direction[r])) / denom — позиція вздовж відрізка (0..1)
    dot_v1_perp = perp_dirs @ v1.T  # (R, S)
    t2 = dot_v1_perp / safe_denom  # (R, S)

    hit = valid & (t1 >= 0.0) & (t2 >= 0.0) & (t2 <= 1.0)
    t1_masked = np.where(hit, t1, np.inf)

    distances = np.min(t1_masked, axis=1)
    return np.minimum(distances, max_distance)


def _closest_center_line_distance(points: np.ndarray, center_line: np.ndarray) -> np.ndarray:
    """Для кожної точки (напр. кута хітбоксу) — відстань до найближчої точки
    center_line. Порівняння з половиною ширини траси визначає, чи точка на
    асфальті. Векторизовано по points; center_line перебирається один раз
    (прийнятно, бо points завжди рівно 4 — кути машини)."""
    result = np.empty(len(points))
    for i, p in enumerate(points):
        result[i] = np.min(np.linalg.norm(center_line - p, axis=1))
    return result


def _build_arc_length_table(center_line: np.ndarray) -> np.ndarray:
    """Кумулятивна довжина дуги center_line, індексована так само, як сама
    center_line (замкнена петля) — рахується ОДИН РАЗ при завантаженні
    траси. arc_length[i] = довжина шляху вздовж дороги від точки 0 до i."""
    diffs = np.diff(center_line, axis=0, append=center_line[:1])
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)[:-1]])


def _track_progress(position: np.ndarray, center_line: np.ndarray, arc_length_table: np.ndarray) -> float:
    """Позиція машини вздовж дороги (довжина дуги до найближчої точки
    center_line) — на відміну від прямої відстані до checkpoint, це НЕ
    заохочує їхати напростець крізь внутрішній кут повороту (де по прямій
    ближче, а по факту там трава/стіна, не асфальт)."""
    nearest_idx = int(np.argmin(np.linalg.norm(center_line - position, axis=1)))
    return float(arc_length_table[nearest_idx])


def _forward_arc_distance(from_arc: float, to_arc: float, track_length: float) -> float:
    """Відстань уздовж дороги ВПЕРЕД від from_arc до to_arc по замкненій
    петлі (завжди >= 0) — якщо ціль позаду за прогресом кола, додає повну
    довжину траси (wrap-around через фініш/старт)."""
    delta = to_arc - from_arc
    return delta if delta >= 0 else delta + track_length


class CarRacingEnv(gym.Env):
    """RL-середовище для одного агента на заданій трасі. Reward-схема узгоджена
    з користувачем: dense potential + checkpoint bonus + швидкість + track
    limits (4 кути хітбоксу, пільговий період на часткове порушення) +
    фінальний бонус за перемогу над референсним часом людини."""

    metadata = {"render_modes": []}

    def __init__(self, track: Track, half_track_width: float = 22.0,
                 reference_time: float | None = None, physics_substeps: int = 1,
                 checkpoint_weight: float = 1.0, speed_weight: float = 1.0,
                 out_of_bounds_weight: float = 1.0):
        super().__init__()
        self.track = track
        self.half_track_width = half_track_width
        self.reference_time = reference_time
        self.physics_substeps = physics_substeps
        self.dt = 1.0 / 60.0

        # Множники компонентів reward — керовані з UI перед стартом навчання
        # (панель налаштувань), 1.0 = базова поведінка, узгоджена раніше.
        self.checkpoint_weight = checkpoint_weight
        self.speed_weight = speed_weight
        self.out_of_bounds_weight = out_of_bounds_weight

        self._seg_p1, self._seg_p2 = _build_edge_segments(track.left_edge, track.right_edge)
        self._arc_length_table = _build_arc_length_table(track.center_line)
        self._track_length = float(self._arc_length_table[-1]) + float(
            np.linalg.norm(track.center_line[-1] - track.center_line[0])
        )
        # Прогрес кожного checkpoint уздовж дороги — рахується ОДИН РАЗ тут
        # (checkpoint_indices — готові індекси в center_line), а не щокроку
        # шукати найближчу точку для самого checkpoint (вона й так нерухома).
        self._checkpoint_arc_lengths = self._arc_length_table[track.checkpoint_indices]

        # Observation: RAY_COUNT промені + швидкість + sin/cos кута до наступного
        # checkpoint (замість сирого кута — уникає розриву на межі -π/π) + відстань.
        obs_dim = RAY_COUNT + 1 + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: [газ(0/1), кермо(STEER_LEVELS), гальмо(BRAKE_LEVELS)]
        self.action_space = spaces.MultiDiscrete([2, STEER_LEVELS, BRAKE_LEVELS])

        self.car: CarState | None = None
        self.next_checkpoint_idx = 0
        self.episode_time = 0.0
        self.laps_completed = 0
        self._prev_arc_distance_to_checkpoint = 0.0
        self._last_rays = np.zeros(RAY_COUNT)

        # --- Стан для нових компонентів reward (стиль водіння) ---
        self._speed_history: deque[tuple[float, float]] = deque()  # (episode_time, speed)
        self._low_speed_timer = 0.0

        # Додатковий штраф за смерть у сегменті, що стабільно вбиває ботів —
        # мапа {segment_idx: extra_penalty}, оновлюється ЗЗОВНІ через
        # set_segment_penalties() (з батьківського процесу, де BackgroundTrainer
        # веде спільну для всіх ботів статистику по сегментах цієї траси).
        self._segment_penalties: dict[int, float] = {}

    def set_segment_penalties(self, penalties: dict[int, float]) -> None:
        self._segment_penalties = penalties

    def set_reference_time(self, reference_time: float | None) -> None:
        """Оновлює референсний час для фінального бонусу кола — викликається
        через env_method(), коли BackgroundTrainer використовує власний
        найкращий час бота як fallback (гравець ще не проходив цю трасу,
        тому reference_time від нього недоступний)."""
        self.reference_time = reference_time

    def set_track(self, track: Track, reference_time: float | None = None) -> None:
        """Підміняє геометрію траси в УЖЕ ІСНУЮЧОМУ середовищі — викликається
        через VecEnv.env_method(), щоб при новій трасі не вбивати й заново
        не піднімати OS-процеси SubprocVecEnv (це дорого на Windows, спричиняло
        зависання вікна на кілька секунд при кожному кліку "Нова траса").

        Примусово переспавнює машину на новому start_pos — інакше бот ще
        кілька кадрів "летів" би по СТАРИХ координатах, які на новій трасі
        можуть опинитись за межами асфальту (і одразу отримати штраф)."""
        self.track = track
        self.reference_time = reference_time
        self._seg_p1, self._seg_p2 = _build_edge_segments(track.left_edge, track.right_edge)
        self._arc_length_table = _build_arc_length_table(track.center_line)
        self._track_length = float(self._arc_length_table[-1]) + float(
            np.linalg.norm(track.center_line[-1] - track.center_line[0])
        )
        self._checkpoint_arc_lengths = self._arc_length_table[track.checkpoint_indices]

        if self.car is not None:
            heading = float(np.arctan2(track.start_dir[1], track.start_dir[0]))
            self.car = CarState(position=track.start_pos.copy(), heading=heading)
            self.next_checkpoint_idx = 1 % len(track.checkpoints)
            self.episode_time = 0.0
            self.laps_completed = 0
            car_arc = _track_progress(self.car.position, track.center_line, self._arc_length_table)
            checkpoint_arc = self._checkpoint_arc_lengths[self.next_checkpoint_idx]
            self._prev_arc_distance_to_checkpoint = _forward_arc_distance(car_arc, checkpoint_arc, self._track_length)

            self._speed_history.clear()
            self._low_speed_timer = 0.0

    def _decode_action(self, action: np.ndarray) -> tuple[int, float, float]:
        throttle = int(action[0])
        steer = float(action[1]) / (STEER_LEVELS - 1) * 2.0 - 1.0
        brake = float(action[2]) / (BRAKE_LEVELS - 1)
        return throttle, steer, brake

    def _get_obs(self) -> np.ndarray:
        assert self.car is not None
        rays = cast_rays_vectorized(self.car.position, self.car.heading, self._seg_p1, self._seg_p2)
        self._last_rays = rays  # кешується для info["render_state"] у step() — не рахувати вдруге
        target = self.track.checkpoints[self.next_checkpoint_idx]
        to_target = target - self.car.position
        dist = np.linalg.norm(to_target)
        angle_to_target = np.arctan2(to_target[1], to_target[0]) - self.car.heading

        obs = np.concatenate([
            rays / RAY_MAX_DISTANCE,
            [self.car.speed / 220.0],
            [np.sin(angle_to_target), np.cos(angle_to_target)],
            [dist / 500.0],
        ]).astype(np.float32)
        return obs

    def _corners_out_of_bounds_count(self) -> int:
        assert self.car is not None
        corners = self.car.corners()
        dists = _closest_center_line_distance(corners, self.track.center_line)
        return int(np.sum(dists > self.half_track_width))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        heading = float(np.arctan2(self.track.start_dir[1], self.track.start_dir[0]))
        self.car = CarState(position=self.track.start_pos.copy(), heading=heading)
        self.next_checkpoint_idx = 1 % len(self.track.checkpoints)
        self.episode_time = 0.0
        self.laps_completed = 0

        car_arc = _track_progress(self.car.position, self.track.center_line, self._arc_length_table)
        checkpoint_arc = self._checkpoint_arc_lengths[self.next_checkpoint_idx]
        self._prev_arc_distance_to_checkpoint = _forward_arc_distance(car_arc, checkpoint_arc, self._track_length)

        self._speed_history.clear()
        self._low_speed_timer = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        assert self.car is not None
        throttle, steer, brake = self._decode_action(action)

        reward = 0.0
        for _ in range(self.physics_substeps):
            self.car = step_car(self.car, throttle=throttle, steer=steer, brake=brake, dt=self.dt)
            self.episode_time += self.dt

        # 1. Dense reward: наближення до наступного checkpoint УЗДОВЖ ДОРОГИ
        # (arc length по center_line), НЕ по прямій лінії. Пряма відстань
        # заохочувала б їхати напростець крізь внутрішній кут гострого
        # повороту (шпильки) — там по прямій найкоротший шлях до checkpoint,
        # хоча фізично там трава/стіна, не асфальт (виявлено користувачем:
        # боти стабільно бились у внутрішній край U-подібного повороту).
        target = self.track.checkpoints[self.next_checkpoint_idx]
        dist = float(np.linalg.norm(target - self.car.position))  # пряма — лише для "чи торкнувся checkpoint"
        car_arc = _track_progress(self.car.position, self.track.center_line, self._arc_length_table)
        checkpoint_arc = self._checkpoint_arc_lengths[self.next_checkpoint_idx]
        arc_dist = _forward_arc_distance(car_arc, checkpoint_arc, self._track_length)
        reward += (self._prev_arc_distance_to_checkpoint - arc_dist)
        self._prev_arc_distance_to_checkpoint = arc_dist

        # 2. Checkpoint bonus.
        terminated = False
        n = len(self.track.checkpoints)
        passed_segment: int | None = None
        just_completed_lap_time: float | None = None
        cycle_completed = False  # True лише коли епізод завершився через MAX_LAPS_PER_EPISODE успіхів, НЕ через виліт
        if dist <= 30.0:
            # Сегмент (поточний next_checkpoint_idx, ДО зарахування) щойно
            # пройдено успішно — окремо від "чи все коло без вильоту в
            # цілому", щоб проблемне поле на конкретному повороті зникало,
            # щойно САМЕ цей поворот стабільно проїжджається, а не лише коли
            # ідеальний весь коло цілком (на трасі з кількома складними
            # поворотами шанс пройти ВСІ одразу мізерний — жоден окремий
            # сегмент інакше ніколи не отримав би "успіх").
            passed_segment = self.next_checkpoint_idx
            self.next_checkpoint_idx = (self.next_checkpoint_idx + 1) % n
            reward += CHECKPOINT_BONUS * self.checkpoint_weight
            new_checkpoint_arc = self._checkpoint_arc_lengths[self.next_checkpoint_idx]
            self._prev_arc_distance_to_checkpoint = _forward_arc_distance(car_arc, new_checkpoint_arc, self._track_length)

            if self.next_checkpoint_idx == 0:
                # Повне коло пройдено — бот НЕ втрачає накопичений reward і
                # їде далі одразу на наступне коло (замість reset), доки не
                # набереться MAX_LAPS_PER_EPISODE поспіль. Це підтримує стимул
                # "берегти й нарощувати очки", а не "закінчити і почати
                # спочатку". Епізод все одно закінчиться природно — або через
                # виліт (track limits нижче), або через ліміт кіл/часу.
                if self.reference_time is not None and self.episode_time > 0:
                    reward += min(1.0, self.reference_time / self.episode_time) * 10.0 * self.checkpoint_weight
                just_completed_lap_time = self.episode_time
                self.laps_completed += 1
                self.episode_time = 0.0  # нове коло рахує свій власний час окремо від попередніх
                if self.laps_completed >= MAX_LAPS_PER_EPISODE:
                    terminated = True
                    cycle_completed = True

        # 2b. Бонус за гальмування перед гострим поворотом. Кут рахується між
        # напрямком (машина→поточний checkpoint) і (поточний→наступний
        # checkpoint) — наскільки різко трасa зламається одразу після цілі,
        # до якої зараз їдемо. Активний лише в межах SHARP_TURN_BRAKE_DISTANCE
        # від checkpoint, щоб не заохочувати гальмувати посеред прямої.
        current_target = self.track.checkpoints[self.next_checkpoint_idx]
        next_target = self.track.checkpoints[(self.next_checkpoint_idx + 1) % n]
        dist_to_current = float(np.linalg.norm(current_target - self.car.position))
        if brake > 0 and dist_to_current <= SHARP_TURN_BRAKE_DISTANCE:
            in_vec = current_target - self.car.position
            out_vec = next_target - current_target
            in_norm = np.linalg.norm(in_vec)
            out_norm = np.linalg.norm(out_vec)
            if in_norm > 1e-6 and out_norm > 1e-6:
                cos_angle = np.clip(np.dot(in_vec, out_vec) / (in_norm * out_norm), -1.0, 1.0)
                turn_angle_deg = np.degrees(np.arccos(cos_angle))
                if turn_angle_deg >= SHARP_TURN_ANGLE_DEG:
                    reward += SHARP_TURN_BRAKE_SCALE * self.car.speed * self.checkpoint_weight

        # 3. Швидкість — невеликий бонус, щоб не тягнути час.
        reward += self.car.speed * SPEED_REWARD_SCALE * self.speed_weight

        # 3b. Штраф за тривалу їзду значно повільніше за власний недавній
        # максимум (ковзне вікно SPEED_WINDOW_SECONDS) — ловить "застрягання",
        # але НЕ карає природне сповільнення в повороті чи розгін після
        # старту (SPEED_LOW_GRACE_SECONDS дає на це пільговий час).
        self._speed_history.append((self.episode_time, self.car.speed))
        while self._speed_history and self.episode_time - self._speed_history[0][0] > SPEED_WINDOW_SECONDS:
            self._speed_history.popleft()
        recent_max_speed = max(s for _, s in self._speed_history)
        if recent_max_speed > 1e-6 and self.car.speed < recent_max_speed * SPEED_LOW_FRACTION:
            self._low_speed_timer += self.dt
            if self._low_speed_timer > SPEED_LOW_GRACE_SECONDS:
                reward += SPEED_LOW_PENALTY * self.speed_weight
        else:
            self._low_speed_timer = 0.0

        # 4. Track limits: 4 кути хітбоксу відносно межі траси. Будь-яке
        # колесо за межею (навіть одне) — миттєва смерть епізоду, без
        # пільгового періоду на apex-cutting (раніше дозволяли 1-3 колеса
        # до OUT_OF_BOUNDS_GRACE_SECONDS — прибрано за проханням користувача).
        #
        # died_segment: "поточний checkpoint" (next_checkpoint_idx НА МОМЕНТ
        # смерті) — сегмент, на якому машина була, коли вилетіла. Звітується
        # в info для BackgroundTrainer, який веде спільну для всіх ботів
        # статистику смертей/проходжень по кожному сегменту й розсилає
        # додатковий segment_penalty назад через set_segment_penalties().
        died_segment = self.next_checkpoint_idx
        out_count = self._corners_out_of_bounds_count()
        segment_result: str | None = None
        result_segment: int | None = None
        if out_count > 0:
            reward += OUT_OF_BOUNDS_PENALTY * self.out_of_bounds_weight
            reward += self._segment_penalties.get(died_segment, 0.0)
            terminated = True
            segment_result = "died"
            result_segment = died_segment
        elif passed_segment is not None:
            # Саме ЦЕЙ сегмент щойно пройдено без вильоту (незалежно від
            # решти кола) — власна "проблемна позначка" на цьому повороті
            # може зникнути, навіть якщо десь далі на трасі бот ще вилітає.
            segment_result = "passed"
            result_segment = passed_segment

        truncated = self.episode_time >= MAX_EPISODE_SECONDS

        obs = self._get_obs()
        info = {
            "lap_completed": just_completed_lap_time is not None,
            "lap_time": just_completed_lap_time,
            "cycle_completed": cycle_completed,
            "segment_result": segment_result,
            "segment_idx": result_segment,
            "render_state": {
                "position": self.car.position.copy(),
                "heading": self.car.heading,
                "rays": self._last_rays,
                "throttle": throttle,
                "steer": steer,
                "brake": brake,
                "current_lap_time": self.episode_time,
                "lap_number": self.laps_completed + 1,
            },
        }
        return obs, reward, terminated, truncated, info
