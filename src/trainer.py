"""Фонове RL-тренування (Етап 4) — PPO вчиться в окремому потоці, поки Arcade-
вікно продовжує малювати кадри в головному потоці. Без цього model.learn()
блокував би event loop і вікно "зависало" б на весь час тренування.

Один "мозок" (одна політика PPO), але N паралельних копій траси (SubprocVecEnv,
кожна у власному OS-процесі) одночасно збирають досвід — це саме те, чому
на екрані видно кілька ботів одразу: вони не змагаються між собою, кожен
проживає свою незалежну спробу на тій самій трасі, а їхній досвід разом
наповнює один спільний rollout buffer, з якого PPO і вчиться.

SubprocVecEnv створюється ОДИН РАЗ, лінькво, всередині фонового потоку при
першому start() — не в конструкторі і не в головному потоці. Спавн N OS-
процесів на Windows повільний (немає fork, кожен процес — холодний старт
нового інтерпретера); створення його синхронно в головному потоці Arcade
заморожувало вікно ("не відповідає") на кілька секунд при кожній новій
трасі. Коли траса міняється, а процеси вже підняті — геометрія оновлюється
через env_method("set_track", ...) в існуючих процесах, без перестворення.

Потоки спілкуються через список RenderSnapshot під м'ютексом: потік навчання
після кожного кроку VecEnv записує туди стан усіх ботів, головний потік читає
його щокадру для рендеру й для лівої панелі з балами.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_env import RAY_ANGLES_DEG, CarRacingEnv
from segment_tracker import SegmentDifficultyTracker
from track_generator import Track

SEGMENT_PENALTY_BROADCAST_INTERVAL = 128  # кроків callback між розсилками env_method (дорога операція)

# Скільки успішних (без вильоту) циклів по MAX_LAPS_PER_EPISODE кіл сумарно
# по ВСІХ ботах разом мають назбиратись, перш ніж програма сама згенерує нову
# трасу — ніби користувач сам натиснув "Нова траса", той самий навчений мозок
# продовжує на новій. Рахуються лише успішні завершення (виліт не рахується).
AUTO_NEW_TRACK_AFTER_CYCLES = 10

# Автозбереження моделі кожні AUTOSAVE_INTERVAL_SECONDS реального часу
# активного навчання — страховка від втрати прогресу при краху/закритті
# вікна. Окрема папка від ручних збережень користувача (logs/models/), щоб
# не змішувати автоматичні знімки з тими, що названі свідомо.
AUTOSAVE_INTERVAL_SECONDS = 30 * 60
AUTOSAVE_DIR = Path(__file__).resolve().parent.parent / "logs" / "models" / "saves"

DEFAULT_BOT_COUNT = 16

_NAME_SYLLABLES_1 = ["Мак", "Рей", "Тор", "Ві", "Зен", "Кор", "Лу", "Не", "Ша", "Ор"]
_NAME_SYLLABLES_2 = ["сім", "бо", "рекс", "літ", "дар", "він", "куб", "нар", "тіс", "лон"]


def _generate_bot_names(count: int) -> list[str]:
    """Випадкові, але детерміновано-унікальні імена ботів (не з реального
    словника — просто щоб у списку не було "Bot 1, Bot 2, ...")."""
    rng = random.Random(1234)
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < count:
        name = rng.choice(_NAME_SYLLABLES_1) + rng.choice(_NAME_SYLLABLES_2)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


@dataclass
class RewardWeights:
    """Ваги reward-функції, які можна крутити в UI перед стартом навчання —
    значення тут лише МНОЖАТЬ базові компоненти з rl_env.py (1.0 = як є)."""
    checkpoint: float = 1.0
    speed: float = 1.0
    out_of_bounds_penalty: float = 1.0


@dataclass
class RenderSnapshot:
    """Стан одного бота на момент останнього кроку — усе, що потрібно Arcade,
    щоб намалювати кадр, рядок у панелі балів, і індикатор натиснутих кнопок
    для виділеного бота (throttle/steer/brake — той самий "пульт", яким
    керує CarRacingEnv.step())."""
    bot_id: int
    name: str
    position: np.ndarray
    heading: float
    ray_distances: np.ndarray
    episode_reward: float = 0.0
    throttle: int = 0
    steer: float = 0.0
    brake: float = 0.0
    current_lap_time: float = 0.0
    lap_number: int = 1
    best_lap_time: float | None = None  # найкращий ВЛАСНИЙ час цього бота (не спільний trainer.best_lap_time)


def _make_env_fn(track: Track, half_track_width: float, reference_time: float | None,
                  reward_weights: RewardWeights):
    def _init():
        return CarRacingEnv(
            track, half_track_width=half_track_width, reference_time=reference_time,
            checkpoint_weight=reward_weights.checkpoint,
            speed_weight=reward_weights.speed,
            out_of_bounds_weight=reward_weights.out_of_bounds_penalty,
        )
    return _init


class BackgroundTrainer:
    """Обгортає PPO + SubprocVecEnv(bot_count копій), навчається у фоновому
    потоці шматками (rollout buffer, n_steps timesteps за раз), не блокуючи
    рендер. speed_multiplier (1x-10x) керує паузою між кроками VecEnv —
    на 1x весь гурт ботів рухається в темпі реального часу, на 10x без
    штучного гальмування.

    VecEnv/PPO створюються ЛІНЬКВО всередині start() (у фоновому потоці) —
    не в __init__ — щоб конструювання BackgroundTrainer (яке відбувається
    при кожній новій трасі) залишалось миттєвим."""

    def __init__(self, track: Track, half_track_width: float = 22.0,
                 reference_time: float | None = None, bot_count: int = DEFAULT_BOT_COUNT,
                 reward_weights: RewardWeights | None = None):
        self.track = track
        self.half_track_width = half_track_width
        self.reference_time = reference_time
        self.bot_count = bot_count
        self.reward_weights = reward_weights or RewardWeights()
        self.bot_names = _generate_bot_names(bot_count)
        self.dt = 1.0 / 60.0

        self.vec_env: SubprocVecEnv | None = None
        self.model: PPO | None = None

        self.speed_multiplier = 1
        self._lock = threading.Lock()
        self._snapshots: list[RenderSnapshot] = [
            RenderSnapshot(
                bot_id=i, name=self.bot_names[i],
                position=track.start_pos.copy(), heading=0.0,
                ray_distances=np.zeros(len(RAY_ANGLES_DEG)),
            )
            for i in range(bot_count)
        ]
        self._episode_reward_acc = [0.0] * bot_count
        self._bot_best_lap_time: list[float | None] = [None] * bot_count
        self._running = False
        self._starting = False
        self._thread: threading.Thread | None = None
        self._pending_load_path: str | None = None

        # Нова траса запитується з ГОЛОВНОГО потоку (клік "Нова траса"), але
        # env_method() на SubprocVecEnv не можна викликати одночасно з тим,
        # як фоновий потік навчання читає з тих самих IPC-каналів усередині
        # model.learn() — дві сторони одночасно шлють/чекають дані на тому ж
        # pipe, і pickle отримує обрізаний буфер (реальний крах, що стався:
        # "pickle data was truncated"). Тому запит лише кладеться в чергу тут,
        # а сам env_method виконується _SnapshotCallback БЕЗПЕЧНО, між кроками
        # env.step(), з того самого потоку, що й усе інше спілкування з VecEnv.
        self._pending_track_update: tuple[Track, float | None] | None = None
        self._track_update_lock = threading.Lock()

        # Спільна для всіх ботів статистика "які сегменти траси стабільно
        # вбивають" — оновлюється в _SnapshotCallback з info кожного бота,
        # штрафи періодично розсилаються назад усім процесам.
        self.segment_tracker = SegmentDifficultyTracker()
        self._segment_count = len(track.checkpoints)

        # Найкращий час кола серед УСІХ ботів разом (не окремо кожного) —
        # для порівняння з рекордом гравця в Leaderboard.
        self.best_lap_time: float | None = None

        # Спільний лічильник успішних циклів (MAX_LAPS_PER_EPISODE кіл поспіль
        # без вильоту) по ВСІХ ботах разом — main.py перевіряє
        # auto_new_track_requested щокадру й сам генерує нову трасу, коли
        # набирається AUTO_NEW_TRACK_AFTER_CYCLES.
        self.cycles_completed = 0
        self._auto_new_track_requested = False

        # Автозбереження — відлік від моменту, коли навчання РЕАЛЬНО почало
        # крутитись (не від конструктора, бо start() може статись значно
        # пізніше). last_autosave_path — для UI/логів, яке ім'я файлу останнє.
        self._last_autosave_time: float | None = None
        self.last_autosave_path: str | None = None

    @property
    def auto_new_track_requested(self) -> bool:
        return self._auto_new_track_requested

    def acknowledge_auto_new_track(self) -> None:
        """main.py викликає одразу після того, як сам згенерував нову трасу
        у відповідь на auto_new_track_requested — скидає прапорець і лічильник."""
        self._auto_new_track_requested = False
        self.cycles_completed = 0

    def get_snapshots(self) -> list[RenderSnapshot]:
        with self._lock:
            return list(self._snapshots)

    def get_problematic_segments(self) -> dict[int, float]:
        """segment_idx -> death_rate (0..1), лише сегменти понад поріг —
        для візуалізації "проблемних полів" на трасі в main.py."""
        return self.segment_tracker.problematic_segments()

    def update_track(self, track: Track, reference_time: float | None = None) -> None:
        """Викликається при генерації нової траси (з ГОЛОВНОГО потоку). Якщо
        VecEnv уже піднятий і активно навчається — сам env_method відкладається
        в чергу й виконується _SnapshotCallback з потоку навчання (безпечно,
        між кроками env.step()). Якщо навчання ще не запущено — просто
        запам'ятовує трасу для майбутнього start()."""
        self.track = track
        self.reference_time = reference_time
        self.segment_tracker = SegmentDifficultyTracker()
        self._segment_count = len(track.checkpoints)
        self.best_lap_time = None
        self._bot_best_lap_time = [None] * self.bot_count
        with self._lock:
            self._snapshots = [
                RenderSnapshot(
                    bot_id=i, name=self.bot_names[i],
                    position=track.start_pos.copy(), heading=0.0,
                    ray_distances=np.zeros(len(RAY_ANGLES_DEG)),
                )
                for i in range(self.bot_count)
            ]
        if self.vec_env is not None:
            if self._running:
                with self._track_update_lock:
                    self._pending_track_update = (track, reference_time)
            else:
                # Навчання не крутиться зараз (наприклад, щойно зупинене) —
                # немає конкуруючого потоку, env_method безпечний одразу.
                self.vec_env.env_method("set_track", track, reference_time)
                self.vec_env.env_method("set_segment_penalties", {})

    @property
    def total_timesteps(self) -> int:
        """Скільки кроків модель РЕАЛЬНО пройшла за все своє життя — на
        відміну від episode_reward у RenderSnapshot (обнуляється щоепізоду),
        це число зберігається і продовжується через PPO.load()/save(), тому
        не "стрибає на нуль" при завантаженні збереженої моделі."""
        return self.model.num_timesteps if self.model is not None else 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_starting(self) -> bool:
        return self._starting

    def start(self) -> None:
        if self._running or self._starting:
            return
        self._starting = True
        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Призупиняє фоновий потік навчання, АЛЕ лишає SubprocVecEnv/модель
        живими — наступний start() продовжує з того самого стану (той самий
        model.num_timesteps, ті самі ваги), не втрачаючи прогрес. Раніше тут
        викликався vec_env.close(), через що "Зупинити" → "Почати навчання"
        завжди намагався писати в уже закритий OS-процес (BrokenPipeError) —
        "продовжити" фізично не могло спрацювати."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._starting = False

    def shutdown(self) -> None:
        """Остаточно закриває SubprocVecEnv (реально вбиває OS-процеси) —
        викликати лише коли цей BackgroundTrainer більше не буде
        використовуватись (заміна на новий трейнер, закриття вікна)."""
        self.stop()
        if self.vec_env is not None:
            self.vec_env.close()
            self.vec_env = None

    def save(self, path: str) -> bool:
        if self.model is None:
            return False
        self.model.save(path)
        return True

    def _maybe_autosave(self) -> None:
        """Викликається з _SnapshotCallback (потік навчання) — не потребує
        env_method/IPC (model.save() працює лише з батьківським об'єктом
        моделі), тому безпечно без черги, на відміну від зміни траси."""
        if self._last_autosave_time is None:
            return
        now = time.perf_counter()
        if now - self._last_autosave_time < AUTOSAVE_INTERVAL_SECONDS:
            return
        self._last_autosave_time = now
        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        path = AUTOSAVE_DIR / f"autosave_{timestamp}.zip"
        if self.save(str(path)):
            self.last_autosave_path = str(path)

    def request_load(self, path: str) -> None:
        """Позначає модель для завантаження при наступному start() — фактичне
        PPO.load() відбувається у фоновому потоці (як і створення VecEnv),
        щоб не блокувати Arcade під час читання файлу з диска."""
        self._pending_load_path = path

    def _train_loop(self) -> None:
        # SubprocVecEnv/PPO піднімаються ТУТ, у фоновому потоці — спавн N
        # OS-процесів займає кілька секунд на Windows, і робити це в
        # головному потоці заморожувало б Arcade-вікно.
        if self.vec_env is None:
            env_fns = [
                _make_env_fn(self.track, self.half_track_width, self.reference_time, self.reward_weights)
                for _ in range(self.bot_count)
            ]
            # Кожен бот — окремий OS-процес (Windows завжди spawn, не fork),
            # який заново імпортує весь torch/CUDA рантайм при старті. Це
            # реальний ліміт пам'яті машини, не щось, що можна обійти
            # прапорцем зсередини коду (CUDA_VISIBLE_DEVICES не встигає
            # подіяти — torch вантажиться ще при імпорті main.py, до
            # виконання будь-якого нашого коду в дочірньому процесі).
            # Тому bot_count у UI свідомо обмежений зверху (Stepper max_value
            # в ui_panel.py) — 50+ ботів валить процес з MemoryError на
            # звичайній машині.
            self.vec_env = SubprocVecEnv(env_fns)
            if self._pending_load_path is not None:
                self.model = PPO.load(self._pending_load_path, env=self.vec_env, device="cpu")
                self._pending_load_path = None
            else:
                self.model = PPO(
                    "MlpPolicy", self.vec_env, verbose=0, device="cpu",
                    n_steps=256, batch_size=256 * self.bot_count // 4 if self.bot_count >= 4 else 64,
                )

        self._starting = False
        self._running = True
        if self._last_autosave_time is None:
            self._last_autosave_time = time.perf_counter()

        callback = _SnapshotCallback(self)
        while self._running:
            self.model.learn(
                total_timesteps=self.model.n_steps * self.bot_count,
                reset_num_timesteps=False,
                callback=callback,
                progress_bar=False,
            )


class _SnapshotCallback(BaseCallback):
    """SB3 викликає on_step() після кожного vec_env.step() (усі bot_count
    середовища одразу) всередині model.learn() — тут публікується знімок
    стану кожного бота з info["render_state"], який CarRacingEnv.step()
    вже заповнює."""

    def __init__(self, trainer: BackgroundTrainer):
        super().__init__(verbose=0)
        self.trainer = trainer
        self._steps_since_broadcast = 0
        self._segment_penalties_dirty = False
        self._reference_time_dirty = False

    def _on_step(self) -> bool:
        trainer = self.trainer

        # Запит на нову трасу з головного потоку (клік "Нова траса") чекав у
        # черзі — виконуємо env_method ТУТ, з потоку навчання, між кроками
        # env.step(), щоб не читати/писати в ті самі IPC-канали одночасно з
        # SubprocVecEnv.step_wait() у collect_rollouts() (саме це раніше
        # валило pickle: "pickle data was truncated").
        pending = None
        with trainer._track_update_lock:
            if trainer._pending_track_update is not None:
                pending = trainer._pending_track_update
                trainer._pending_track_update = None
        if pending is not None:
            new_track, new_reference_time = pending
            trainer.vec_env.env_method("set_track", new_track, new_reference_time)
            trainer.vec_env.env_method("set_segment_penalties", {})

        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if infos is not None:
            new_snapshots = list(trainer._snapshots)
            for i, info in enumerate(infos):
                render_state = info.get("render_state")
                if render_state is None:
                    continue

                if rewards is not None:
                    trainer._episode_reward_acc[i] += float(rewards[i])
                if dones is not None and bool(dones[i]):
                    trainer._episode_reward_acc[i] = 0.0

                # Спільна статистика "які сегменти вбивають" — оновлюється тут
                # (батьківський процес), а не в CarRacingEnv (де кожен бот бачив
                # би лише свою власну історію, не всіх ботів разом). Рахується
                # per-сегмент: "died"/"passed" стосуються ЛИШЕ segment_idx, не
                # всієї траси — так проблемна позначка на одному повороті може
                # зникнути, навіть якщо десь далі бот ще вилітає.
                segment_result = info.get("segment_result")
                segment_idx = info.get("segment_idx")
                if segment_result == "died" and segment_idx is not None:
                    trainer.segment_tracker.record_death(segment_idx)
                    self._segment_penalties_dirty = True
                elif segment_result == "passed" and segment_idx is not None:
                    trainer.segment_tracker.record_pass(segment_idx)
                    self._segment_penalties_dirty = True

                lap_time = info.get("lap_time")
                if lap_time is not None:
                    if trainer._bot_best_lap_time[i] is None or lap_time < trainer._bot_best_lap_time[i]:
                        trainer._bot_best_lap_time[i] = lap_time
                    if trainer.best_lap_time is None or lap_time < trainer.best_lap_time:
                        trainer.best_lap_time = lap_time
                        # Якщо гравець ще не проходив цю трасу (reference_time
                        # від нього недоступний) — власний найкращий час бота
                        # стає fallback-орієнтиром для фінального бонусу кола,
                        # інакше бот тренується без стимулу "бути швидшим".
                        if trainer.reference_time is None:
                            self._reference_time_dirty = True

                # Спільний лічильник успішних циклів (MAX_LAPS_PER_EPISODE
                # кіл поспіль без вильоту) по ВСІХ ботах разом — коли
                # набирається AUTO_NEW_TRACK_AFTER_CYCLES, просимо main.py
                # згенерувати нову трасу самостійно (auto_new_track_requested).
                if info.get("cycle_completed"):
                    trainer.cycles_completed += 1
                    if trainer.cycles_completed >= AUTO_NEW_TRACK_AFTER_CYCLES:
                        trainer._auto_new_track_requested = True

                new_snapshots[i] = RenderSnapshot(
                    bot_id=i, name=trainer.bot_names[i],
                    position=render_state["position"],
                    heading=render_state["heading"],
                    ray_distances=render_state["rays"],
                    episode_reward=trainer._episode_reward_acc[i],
                    throttle=render_state["throttle"],
                    steer=render_state["steer"],
                    brake=render_state["brake"],
                    current_lap_time=render_state["current_lap_time"],
                    lap_number=render_state["lap_number"],
                    best_lap_time=trainer._bot_best_lap_time[i],
                )

            with trainer._lock:
                trainer._snapshots = new_snapshots

        # Розсилка оновлених штрафів усім процесам — не щокроку (env_method
        # це IPC-виклик у КОЖЕН з bot_count процесів, дорого при 20+ ботах),
        # а раз на SEGMENT_PENALTY_BROADCAST_INTERVAL кроків, і лише якщо
        # статистика справді змінилась відтоді.
        self._steps_since_broadcast += 1
        if self._steps_since_broadcast >= SEGMENT_PENALTY_BROADCAST_INTERVAL:
            if self._segment_penalties_dirty:
                penalties = trainer.segment_tracker.penalties()
                trainer.vec_env.env_method("set_segment_penalties", penalties)
                self._segment_penalties_dirty = False
            if self._reference_time_dirty:
                trainer.vec_env.env_method("set_reference_time", trainer.best_lap_time)
                self._reference_time_dirty = False
            self._steps_since_broadcast = 0

        trainer._maybe_autosave()

        speed = max(1, trainer.speed_multiplier)
        pause = trainer.dt / speed
        if pause > 0.0005:
            time.sleep(pause)

        return trainer._running
