"""Гоночне навчання (Етап 6, мультиагент): 8 машин в ОДНОМУ спільному світі,
стартова сітка як у Ф1, фізичні колізії між машинами, місця/очки за фініш
і ротація стартової сітки між гонками.

Чому не SubprocVecEnv: колізії вимагають, щоб усі машини жили в одному
фізичному світі — їхні кроки не можна рахувати незалежно у 8 окремих
OS-процесах. Тому тут власний VecEnv (RaceVecEnv) в ОДНОМУ процесі: PPO
бачить звичні 8 паралельних середовищ, а під капотом це 8 "поглядів" на
один спільний заїзд (self-play: одна політика керує всіма машинами).
Бонус: зникає IPC між процесами — швидше і без пов'язаних мікрофризів.

Механіка гонки:
- Всі 8 стартують колонами по двоє за стартовою лінією, ціль — 3 кола
  (MAX_LAPS_PER_EPISODE) якнайшвидше.
- Смерть (виліт/рахунок) не перериває гонку для інших: мертва машина
  "замерзає" на місці аварії, отримує 8-ме місце і нульові нагороди до
  кінця заїзду. Епізоди всіх 8 закінчуються ОДНОЧАСНО з фінішем гонки —
  так rollout buffer PPO отримує рівні траєкторії.
- Очки за місця (RACE_POINTS × race_points_scale з налаштувань) додаються
  терминальним бонусом в останній крок гонки.
- Наступна стартова сітка: місце = 9 − середнє(старт, фініш) — формула
  користувача: виграв з поулу → стартуєш останнім (вчися обганяти),
  провалився з поулу → стартуєш посередині.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from car import CAR_LENGTH, CarState, step_car
from rl_env import CarRacingEnv
from settings import RewardConfig
from track_generator import Track

NUM_RACE_CARS = 8
RACE_TIME_LIMIT_SECONDS = 180.0  # страховка: гонка не триває вічно, якщо ніхто не фінішує

# Очки за місця 1..8 — профіль за проханням користувача: 7-8 нічого,
# 5-6 трошки, далі наростає. Масштабується race_points_scale з налаштувань.
RACE_POINTS = [10.0, 7.0, 5.0, 3.0, 1.5, 1.0, 0.0, 0.0]

# Стартова сітка: колони по двоє (як у Ф1), відстань між рядами вздовж
# траси і бічний зсув від осі. Ширина траси ~44 (half_track_width 22),
# машина 14×7 — два стовпці з зсувом ±6.5 вміщаються з запасом.
GRID_ROW_GAP = 22.0
GRID_LATERAL_OFFSET = 6.5

# Колізії: корпус апроксимується колом (для 8 машин точність прямокутників
# не варта складності). Контактна відстань 2R=11 — трохи менша за довжину
# машини (14): легке візуальне перекриття носа з хвостом прийнятне для
# аркадного стилю. Пружність 0.3 — "потерлись і роз'їхались", не більярд.
COLLISION_RADIUS = 5.5
COLLISION_RESTITUTION = 0.3

# Накат після фінішу: без газу, drag гальмує природно; зупинка при цій
# швидкості або по таймауту (страховка, якщо машина покотилась з гірки).
FINISH_COAST_STOP_SPEED = 15.0
FINISH_COAST_MAX_SECONDS = 4.0


def _resolve_collisions(states: list[CarState | None]) -> None:
    """Розштовхує машини, що перетнулись (позиційно) + обмін імпульсом
    уздовж лінії контакту. БЕЗ штучного гальмування чи штрафів — звичайна
    "ігрова" фізика зіткнень (запит користувача: екшн — це весело).
    Мутує position/velocity прямо в CarState (це numpy-масиви)."""
    n = len(states)
    for i in range(n):
        si = states[i]
        if si is None:
            continue
        for j in range(i + 1, n):
            sj = states[j]
            if sj is None:
                continue
            diff = sj.position - si.position
            dist = float(np.linalg.norm(diff))
            if dist >= COLLISION_RADIUS * 2:
                continue
            if dist < 1e-6:
                # Точне накладання (телепорт на старті) — розводимо по X.
                diff = np.array([1.0, 0.0])
                dist = 1e-6
            normal = diff / dist

            overlap = COLLISION_RADIUS * 2 - dist
            si.position -= normal * (overlap / 2)
            sj.position += normal * (overlap / 2)

            # Імпульс лише якщо зближаються (інакше вони вже роз'їжджаються).
            rel_v = sj.velocity - si.velocity
            approach = float(np.dot(rel_v, normal))
            if approach < 0:
                impulse = -(1 + COLLISION_RESTITUTION) * approach / 2
                si.velocity -= normal * impulse
                sj.velocity += normal * impulse


class RaceWorld:
    """Спільний світ гонки: 8 CarRacingEnv (нагороди/checkpoint-и/промені
    кожної машини), але фізика всіх машин рахується ТУТ разом з колізіями,
    а в env-и передається через external_step()."""

    def __init__(self, track: Track, config: RewardConfig | None = None,
                 num_cars: int = NUM_RACE_CARS, half_track_width: float = 22.0,
                 reference_time: float | None = None, physics_substeps: int = 4):
        self.num_cars = num_cars
        self.physics_substeps = physics_substeps
        self.config = config or RewardConfig()
        self.envs = [
            CarRacingEnv(track, half_track_width=half_track_width,
                         reference_time=reference_time, physics_substeps=physics_substeps,
                         config=self.config)
            for _ in range(num_cars)
        ]
        self.track = track
        # Стартова сітка: grid_slots[k] = індекс машини, що стартує з
        # позиції k+1. Перша гонка — просто по порядку.
        self.grid_slots = list(range(num_cars))
        self._reset_race_state()

    # --- Стан гонки ---

    def _reset_race_state(self) -> None:
        self.race_time = 0.0
        self.finished_place: list[int | None] = [None] * self.num_cars  # місце фінішера (1..N)
        self.dead: list[bool] = [False] * self.num_cars
        self._frozen_state: list[CarState | None] = [None] * self.num_cars
        # Фінішери не зупиняються НА лінії (заважали б тим, хто ззаду) —
        # котяться вперед накатом (без газу, drag сам гальмує), поки не
        # сповільняться або не мине ліміт часу. car_idx -> час накату.
        self._coasting: dict[int, float] = {}
        self._next_finish_place = 1
        # стартова позиція кожної машини в ЦІЙ гонці (1..N) — для ротації
        self.start_positions = [0] * self.num_cars
        for slot, car_idx in enumerate(self.grid_slots):
            self.start_positions[car_idx] = slot + 1

    def _grid_pose(self, slot: int) -> tuple[np.ndarray, float]:
        """Світова позиція стартового слота (0 = поул) — колони по двоє
        позаду стартової лінії."""
        start = self.track.start_pos
        direction = self.track.start_dir
        normal = np.array([-direction[1], direction[0]])
        row = slot // 2
        side = 1.0 if slot % 2 == 0 else -1.0
        # Перший ряд трохи позаду лінії, щоб фініш зараховувався перетином.
        pos = start - direction * (CAR_LENGTH + row * GRID_ROW_GAP) + normal * (side * GRID_LATERAL_OFFSET)
        heading = float(np.arctan2(direction[1], direction[0]))
        return pos, heading

    def reset_race(self) -> np.ndarray:
        """Нова гонка: env-и скидаються, машини розставляються по сітці."""
        self._reset_race_state()
        obs = []
        for car_idx in range(self.num_cars):
            self.envs[car_idx].reset()
        for slot, car_idx in enumerate(self.grid_slots):
            pos, heading = self._grid_pose(slot)
            self.envs[car_idx].place_car(pos, heading)
        self._update_ray_obstacles()
        for car_idx in range(self.num_cars):
            obs.append(self.envs[car_idx]._get_obs())
        return np.stack(obs)

    def set_track(self, track: Track, reference_time: float | None = None) -> None:
        """Нова траса: сітка скидається до порядку за замовчуванням (стара
        ротація втрачає сенс — інша траса, інші сили)."""
        self.track = track
        for env in self.envs:
            env.set_track(track, reference_time)
        self.grid_slots = list(range(self.num_cars))
        self.reset_race()

    # --- Допоміжне ---

    def _update_ray_obstacles(self) -> None:
        """Кожній машині — корпуси УСІХ ІНШИХ живих машин як відрізки для
        променів. Мертві (замерзлі) теж перешкода: розбита машина на трасі
        фізично існує і її треба об'їжджати."""
        corners = []
        for env in self.envs:
            corners.append(env.car.corners() if env.car is not None else None)
        for i, env in enumerate(self.envs):
            p1_list, p2_list = [], []
            for j, c in enumerate(corners):
                if j == i or c is None:
                    continue
                p1_list.append(c)
                p2_list.append(np.roll(c, -1, axis=0))
            if p1_list:
                env.set_extra_ray_segments(np.concatenate(p1_list), np.concatenate(p2_list))
            else:
                env.set_extra_ray_segments(None, None)

    def _live_positions(self) -> list[int]:
        """Поточні місця (1..N) для HUD і фінального ранжування на таймауті:
        фінішери — за порядком фінішу, живі — за прогресом (кола + дуга),
        мертві — останні (всі "8-мі", за словами користувача)."""
        progress = []
        n_checkpoints = len(self.track.checkpoints)
        for i, env in enumerate(self.envs):
            if self.finished_place[i] is not None:
                key = (0, self.finished_place[i], 0, 0.0)  # фінішери попереду, за місцем
            elif not self.dead[i]:
                # Прогрес: кола → пройдені checkpoint-и в колі → відстань до
                # наступного (менша = далі). Від'ємні, бо сортуємо за зростанням.
                passed_in_lap = (env.next_checkpoint_idx - 1) % n_checkpoints
                key = (1, -env.laps_completed, -passed_in_lap, env._prev_arc_distance_to_checkpoint)
            else:
                key = (2, 0, 0, 0.0)  # мертві в кінці
            progress.append((key, i))
        progress.sort(key=lambda t: t[0])
        places = [0] * self.num_cars
        for place, (_, i) in enumerate(progress, start=1):
            places[i] = place
        return places

    def _race_over(self) -> bool:
        all_terminal = all(
            self.finished_place[i] is not None or self.dead[i] for i in range(self.num_cars)
        )
        return all_terminal or self.race_time >= RACE_TIME_LIMIT_SECONDS

    def _final_places(self) -> list[int]:
        """Фінальні місця: фінішери за порядком, живі-нефінішери за прогресом.
        МЕРТВІ — завжди 8-мі (точніше, останнє місце): очки їм не світять
        незалежно від того, як далеко заїхали до аварії."""
        live = self._live_positions()
        places = list(live)
        for i in range(self.num_cars):
            if self.dead[i]:
                places[i] = self.num_cars
        return places

    def _rotate_grid(self, final_places: list[int]) -> None:
        """Формула користувача: наступний старт = 9 − середнє(старт, фініш).
        Виграв з поулу (1,1) → 8-й старт; провалився з поулу (1,8) → ~4-5."""
        desired = []
        for i in range(self.num_cars):
            value = (self.num_cars + 1) - (self.start_positions[i] + final_places[i]) / 2
            desired.append((value, i))
        desired.sort(key=lambda t: t[0])
        self.grid_slots = [i for _, i in desired]

    # --- Головний крок ---

    def step(self, actions: np.ndarray):
        """Один крок усіх 8 машин. Повертає (obs, rewards, dones, infos) у
        форматі VecEnv. done для всіх — ОДНОЧАСНО, в кінці гонки."""
        num = self.num_cars
        rewards = np.zeros(num, dtype=np.float32)
        infos: list[dict] = [{} for _ in range(num)]

        # 1. Фізика всіх живих машин разом, з колізіями на кожному сабстепі.
        # Фінішери в накаті (coasting) теж рухаються — без газу, drag гальмує.
        active = [i for i in range(num)
                  if not self.dead[i] and self.finished_place[i] is None]
        coasting = list(self._coasting.keys())
        decoded = {i: self.envs[i]._decode_action(actions[i]) for i in active}
        for i in coasting:
            # Накат: без газу й керма, з легким гальмом — щоб від'їхати від
            # лінії достатньо (не заважати тим, хто фінішує позаду), але не
            # котитися пів траси (чистий drag зупиняв аж за ~300 одиниць).
            decoded[i] = (0, 0.0, 0.4)
        states: list[CarState | None] = [None] * num
        for i in active:
            states[i] = self.envs[i].car
        for i in coasting:
            states[i] = self._frozen_state[i]
        # Мертві/зупинені машини — нерухомі перешкоди в колізіях.
        for i in range(num):
            if states[i] is None and self._frozen_state[i] is not None:
                states[i] = self._frozen_state[i]

        dt = self.envs[0].dt
        moving = active + coasting
        for _ in range(self.physics_substeps):
            for i in moving:
                throttle, steer, brake = decoded[i]
                states[i] = step_car(states[i], throttle=throttle, steer=steer, brake=brake, dt=dt)
            _resolve_collisions(states)

        self.race_time += dt * self.physics_substeps

        # Оновлення накату: позиція рухається за фізикою, зупинка при
        # повільній швидкості або по таймауту.
        for i in coasting:
            self._frozen_state[i] = states[i]
            self._coasting[i] += dt * self.physics_substeps
            if states[i].speed < FINISH_COAST_STOP_SPEED or self._coasting[i] > FINISH_COAST_MAX_SECONDS:
                del self._coasting[i]

        # 2. Нагороди/checkpoint-и/track limits — через external_step кожного env.
        self._update_ray_obstacles_from_states(states)
        for i in active:
            obs_i, reward_i, terminated, truncated, info_i = self.envs[i].external_step(states[i], actions[i])
            rewards[i] = reward_i
            infos[i] = info_i
            if terminated or truncated:
                if info_i.get("cycle_completed"):
                    # Фініш 3 кіл — місце за порядком перетину лінії. Машина
                    # не зупиняється на лінії, а котиться далі накатом.
                    self.finished_place[i] = self._next_finish_place
                    self._next_finish_place += 1
                    self._coasting[i] = 0.0
                else:
                    self.dead[i] = True
                self._frozen_state[i] = self.envs[i].car

        # 3. Замерзлі: нульова нагорода, порожній obs, снапшот на місці аварії.
        for i in range(num):
            if i in active:
                continue
            env = self.envs[i]
            frozen = self._frozen_state[i]
            infos[i] = self._frozen_info(env, frozen)

        # 4. Живі позиції в info для HUD.
        live_places = self._live_positions()
        for i in range(num):
            if "render_state" in infos[i]:
                infos[i]["render_state"]["race_position"] = live_places[i]

        race_over = self._race_over()
        dones = np.full(num, race_over, dtype=bool)

        if race_over:
            final_places = self._final_places()
            for i in range(num):
                place_idx = min(final_places[i], len(RACE_POINTS)) - 1
                rewards[i] += RACE_POINTS[place_idx] * self.config.race_points_scale
                infos[i]["race_final_place"] = final_places[i]
            self._rotate_grid(final_places)
            # terminal_observation — стандарт VecEnv: справжній останній obs
            # епізоду, бо основний масив вже міститиме obs НОВОЇ гонки.
            terminal_obs = self._collect_obs()
            for i in range(num):
                infos[i]["terminal_observation"] = terminal_obs[i]
                # Гонка обірвана таймаутом (не природний кінець для живих) —
                # PPO має bootstrap-ити value, а не вважати це смертю.
                if (self.race_time >= RACE_TIME_LIMIT_SECONDS
                        and self.finished_place[i] is None and not self.dead[i]):
                    infos[i]["TimeLimit.truncated"] = True
            obs = self.reset_race()
        else:
            obs = self._collect_obs()

        return obs, rewards, dones, infos

    def _update_ray_obstacles_from_states(self, states: list[CarState | None]) -> None:
        """Як _update_ray_obstacles, але з щойно порахованих станів (env.car
        ще не оновлені external_step-ом на момент виклику)."""
        corners = [s.corners() if s is not None else None for s in states]
        for i, env in enumerate(self.envs):
            p1_list, p2_list = [], []
            for j, c in enumerate(corners):
                if j == i or c is None:
                    continue
                p1_list.append(c)
                p2_list.append(np.roll(c, -1, axis=0))
            if p1_list:
                env.set_extra_ray_segments(np.concatenate(p1_list), np.concatenate(p2_list))
            else:
                env.set_extra_ray_segments(None, None)

    def _collect_obs(self) -> np.ndarray:
        obs = []
        for i, env in enumerate(self.envs):
            if self.dead[i] or self.finished_place[i] is not None:
                # Замерзла машина "нічого не бачить" — нульовий вектор, як
                # і промені при смерті: нейтральні дані до кінця гонки.
                obs.append(np.zeros(env.observation_space.shape, dtype=np.float32))
            else:
                obs.append(env._get_obs())
        return np.stack(obs)

    def _frozen_info(self, env: CarRacingEnv, frozen: CarState | None) -> dict:
        position = frozen.position if frozen is not None else self.track.start_pos
        heading = frozen.heading if frozen is not None else 0.0
        return {
            "lap_completed": False,
            "lap_time": None,
            "cycle_completed": False,
            "segment_result": None,
            "segment_idx": None,
            "render_state": {
                "position": position.copy() if hasattr(position, "copy") else position,
                "heading": heading,
                "rays": np.zeros(len(env.ray_angles_deg)),
                "throttle": 0,
                "steer": 0.0,
                "brake": 0.0,
                "current_lap_time": env.episode_time,
                "lap_number": env.laps_completed + 1,
            },
        }


class RaceVecEnv(VecEnv):
    """Мінімальна VecEnv-обгортка над RaceWorld для stable-baselines3 —
    PPO працює з нею так само, як із SubprocVecEnv, але все в одному процесі."""

    def __init__(self, track: Track, config: RewardConfig | None = None,
                 num_cars: int = NUM_RACE_CARS, half_track_width: float = 22.0,
                 reference_time: float | None = None, physics_substeps: int = 4):
        self.world = RaceWorld(
            track, config=config, num_cars=num_cars, half_track_width=half_track_width,
            reference_time=reference_time, physics_substeps=physics_substeps,
        )
        env0 = self.world.envs[0]
        super().__init__(num_cars, env0.observation_space, env0.action_space)
        self._actions: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        return self.world.reset_race()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None
        return self.world.step(self._actions)

    def close(self) -> None:
        pass  # немає ні процесів, ні ресурсів для звільнення

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        """Трансляція командних методів. set_track перехоплюється світом
        (треба скинути гонку/сітку цілісно), решта — кожному env."""
        if method_name == "set_track":
            self.world.set_track(*args, **kwargs)
            return [None] * self.num_envs
        if method_name == "set_reward_config":
            self.world.config = args[0]
        results = []
        for env in self.world.envs:
            results.append(getattr(env, method_name)(*args, **kwargs))
        return results

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(env, attr_name) for env in self.world.envs]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        for env in self.world.envs:
            setattr(env, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
