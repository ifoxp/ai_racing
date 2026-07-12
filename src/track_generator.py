"""
Процедурна генерація гоночної траси за seed.

Пайплайн (перевірений на 100+ прогонах у навчальному прототипі,
див. Навчання/02-track-generation.html):

    точки (seed) -> Delaunay -> MST (Prim) -> DFS-обхід дерева в цикл
    -> 2-opt (усунення самоперетинів) -> Catmull-Rom (згладжування)
    -> офсетні контури (ширина асфальту)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay


@dataclass
class Track:
    seed: int
    center_line: np.ndarray  # (N, 2) — згладжена замкнена центральна лінія
    left_edge: np.ndarray    # (N, 2) — лівий край асфальту
    right_edge: np.ndarray   # (N, 2) — правий край асфальту
    checkpoints: np.ndarray  # (M, 2) — контрольні точки вздовж центральної лінії
    control_points: np.ndarray  # (K, 2) — вихідні точки циклу до згладжування
    start_pos: np.ndarray    # (2,) — координати центру стартової лінії
    start_dir: np.ndarray    # (2,) — нормалізований вектор напрямку руху на старті


def _sample_points(rng: np.random.Generator, count: int, radius: float,
                    min_gap: float | None = None, max_attempts: int = 200) -> np.ndarray:
    """Розкидати точки по колу з випадковим офсетом радіуса — уникає самоперетинів
    траси за побудовою (кожен кут сектора займає рівно одна точка).

    min_gap задає мінімальну відстань між БУДЬ-ЯКИМИ двома точками (не тільки сусідніми
    по куту) — без цього дві точки на різних, непослідовних ділянках кола можуть опинитись
    фізично поруч, і асфальт траси на цих ділянках накладається сам на себе (Poisson disk
    sampling — та сама ідея, яку користувач сформулював ще на етапі планування генерації)."""
    if min_gap is None:
        min_gap = radius * 0.22

    for _ in range(max_attempts):
        angles = np.sort(rng.uniform(0, 2 * np.pi, count))
        radii = radius * (0.65 + rng.uniform(0, 0.55, count))
        xs = np.cos(angles) * radii
        ys = np.sin(angles) * radii
        points = np.column_stack([xs, ys])

        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        if dist.min() >= min_gap:
            return points

    # Останній прогін використовується як fallback навіть якщо min_gap не досягнуто —
    # краще трохи щільніша траса, ніж нескінченний цикл спроб.
    return points


def _build_delaunay_edges(points: np.ndarray) -> set[tuple[int, int]]:
    tri = Delaunay(points)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edges.add((a, b) if a < b else (b, a))
    return edges


def _prim_mst(points: np.ndarray, edges: set[tuple[int, int]]) -> list[list[int]]:
    n = len(points)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b in edges:
        w = float(np.linalg.norm(points[a] - points[b]))
        adjacency[a].append((b, w))
        adjacency[b].append((a, w))

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    mst_adj: list[list[int]] = [[] for _ in range(n)]
    remaining = n - 1

    while remaining > 0:
        best_weight = float("inf")
        best_pair: tuple[int, int] | None = None
        for u in range(n):
            if not in_tree[u]:
                continue
            for v, w in adjacency[u]:
                if not in_tree[v] and w < best_weight:
                    best_weight = w
                    best_pair = (u, v)
        if best_pair is None:
            raise RuntimeError("Delaunay-граф виявився незв'язним — це не мало статись")
        u, v = best_pair
        in_tree[v] = True
        mst_adj[u].append(v)
        mst_adj[v].append(u)
        remaining -= 1

    return mst_adj


def _tree_to_cycle(mst_adj: list[list[int]], root: int = 0) -> list[int]:
    """DFS-обхід дерева з shortcutting: кожна вершина відвідується рівно раз,
    що дає гамільтонів цикл по дереву (наближений розв'язок TSP)."""
    n = len(mst_adj)
    visited = np.zeros(n, dtype=bool)
    order: list[int] = []
    stack = [root]
    visited[root] = True
    while stack:
        u = stack.pop()
        order.append(u)
        for v in sorted(mst_adj[u], reverse=True):
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    return order


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    return ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4))


def _ccw_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (c[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _find_crossing_with_edge_i(points: np.ndarray, cycle_arr: np.ndarray, i: int) -> int | None:
    """Векторизовано перевіряє ребро i проти всіх ребер j >= i+2 одним numpy-проходом.
    Повертає перший знайдений j або None."""
    n = len(cycle_arr)
    a1, a2 = points[cycle_arr[i]], points[cycle_arr[(i + 1) % n]]

    j_start = i + 2
    j_end = n if i > 0 else n - 1  # уникнути пари (0, n-1) — суміжні ребра циклу
    if j_start >= j_end:
        return None

    js = np.arange(j_start, j_end)
    b1 = points[cycle_arr[js]]
    b2 = points[cycle_arr[(js + 1) % n]]

    d1 = _ccw_vec(np.broadcast_to(b1, b1.shape), np.broadcast_to(b2, b2.shape), a1)
    d2 = _ccw_vec(b1, b2, a2)
    d3 = _ccw_vec(np.broadcast_to(a1, b1.shape), np.broadcast_to(a2, b1.shape), b1)
    d4 = _ccw_vec(np.broadcast_to(a1, b1.shape), np.broadcast_to(a2, b1.shape), b2)

    cond1 = ((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0))
    cond2 = ((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0))
    hits = np.nonzero(cond1 & cond2)[0]
    if len(hits) == 0:
        return None
    return int(js[hits[0]])


def _remove_crossings(points: np.ndarray, cycle: list[int], max_iterations: int = 2000) -> list[int]:
    """2-opt: поки в циклі є перетин двох ребер — розвернути ділянку між ними.
    Перевірено на 100 випадкових наборах точок (JS-прототип і Python-версія): 0 самоперетинів
    після виправлення. Векторизовано через numpy — перевірка ребра проти всіх інших одним проходом
    замість вкладеного Python-циклу (критично для щільної polyline після згладжування, ~300 точок)."""
    cycle_arr = np.array(cycle, dtype=np.int64)
    n = len(cycle_arr)
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(n):
            j = _find_crossing_with_edge_i(points, cycle_arr, i)
            if j is not None:
                cycle_arr[i + 1 : j + 1] = cycle_arr[i + 1 : j + 1][::-1]
                improved = True
                break
    return cycle_arr.tolist()


def _centripetal_catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, samples: int) -> np.ndarray:
    """Обчислює доцентровий (Centripetal) Catmull-Rom сплайн між p1 та p2.
    Усуває утворення петель (overshooting), які виникають у звичайному Uniform сплайні."""
    def get_t(t_start, p_start, p_end, alpha=0.5):
        d = np.linalg.norm(p_end - p_start)
        if d == 0:
            return t_start
        return t_start + d**alpha

    t0 = 0.0
    t1 = get_t(t0, p0, p1)
    t2 = get_t(t1, p1, p2)
    t3 = get_t(t2, p2, p3)

    if t1 == t2:
        return np.tile(p1, (samples, 1))

    t = np.linspace(t1, t2, samples, endpoint=False)

    A1 = (t1 - t)[:, None] / (t1 - t0) * p0 + (t - t0)[:, None] / (t1 - t0) * p1
    A2 = (t2 - t)[:, None] / (t2 - t1) * p1 + (t - t1)[:, None] / (t2 - t1) * p2
    A3 = (t3 - t)[:, None] / (t3 - t2) * p2 + (t - t2)[:, None] / (t3 - t2) * p3

    B1 = (t2 - t)[:, None] / (t2 - t0) * A1 + (t - t0)[:, None] / (t2 - t0) * A2
    B2 = (t3 - t)[:, None] / (t3 - t1) * A2 + (t - t1)[:, None] / (t3 - t1) * A3

    C = (t2 - t)[:, None] / (t2 - t1) * B1 + (t - t1)[:, None] / (t2 - t1) * B2
    return C


def _chaikin_cut(points: np.ndarray, iterations: int = 2, ratio: float = 0.25) -> np.ndarray:
    """Алгоритм Чайкіна для зрізання кутів. Зберігає унікальну форму траси, але
    зрізає гострі кути, запобігаючи перекручуванню нормалей (країв асфальту)."""
    curr_points = points.copy()
    for _ in range(iterations):
        n = len(curr_points)
        new_points = []
        for i in range(n):
            p0 = curr_points[i]
            p1 = curr_points[(i + 1) % n]
            q = (1 - ratio) * p0 + ratio * p1
            r = ratio * p0 + (1 - ratio) * p1
            new_points.append(q)
            new_points.append(r)
        curr_points = np.array(new_points)
    return curr_points


def _smooth_dense_line(points: np.ndarray, passes: int = 3) -> np.ndarray:
    """Пост-обробка: згладжує фінальну щільну лінію (Moving Average). Зберігає
    макро-форму траси, але повністю 'розплутує' мікро-вузли та різкі злами,
    що виникають через математичні аномалії сплайну."""
    smoothed = points.copy()
    for _ in range(passes):
        prev_pts = np.roll(smoothed, 1, axis=0)
        next_pts = np.roll(smoothed, -1, axis=0)
        smoothed = 0.25 * prev_pts + 0.5 * smoothed + 0.25 * next_pts
    return smoothed


def _smooth_closed_loop(points: np.ndarray, samples_per_segment: int = 16) -> np.ndarray:
    n = len(points)
    segments = []
    for i in range(n):
        p0, p1, p2, p3 = (
            points[(i - 1) % n],
            points[i],
            points[(i + 1) % n],
            points[(i + 2) % n],
        )
        segments.append(_centripetal_catmull_rom(p0, p1, p2, p3, samples_per_segment))
    return np.concatenate(segments, axis=0)


def _offset_loop(points: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    tangents = next_pts - prev_pts
    lengths = np.linalg.norm(tangents, axis=1)
    lengths[lengths == 0] = 1.0
    tangents = tangents / lengths[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    left = points + normals * width
    right = points - normals * width
    return left, right


def _is_track_valid(center_line: np.ndarray, track_width: float) -> bool:
    """Векторизована перевірка на фізичне накладання ділянок траси. Перевіряє,
    чи є будь-які дві віддалені (по індексу) точки траси ближчими одна до одної,
    ніж повна ширина асфальту."""
    step = 4
    pts = center_line[::step]
    m = len(pts)

    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    idx = np.arange(m)
    idx_diff = np.abs(idx[:, None] - idx[None, :])
    cycle_dist = np.minimum(idx_diff, m - idx_diff)

    ignore_window = max(5, int(m * 0.15))
    dist[cycle_dist < ignore_window] = np.inf

    min_clearance = track_width * 2.2

    return bool(np.min(dist) >= min_clearance)


def _has_sharp_turns(points: np.ndarray, min_angle_deg: float = 75.0) -> bool:
    """Векторизовано обчислює внутрішні кути між усіма сусідніми відрізками.
    Повертає True, якщо є хоча б один кут, гостріший за min_angle_deg. Це
    запобігає утворенню вузьких 'шпильок' (hairpins)."""
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)

    v1 = prev_pts - points
    v2 = next_pts - points

    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)

    valid = (n1 > 0) & (n2 > 0)
    if not np.any(valid):
        return False

    v1_norm = v1[valid] / n1[valid, None]
    v2_norm = v2[valid] / n2[valid, None]

    dot_prod = np.sum(v1_norm * v2_norm, axis=-1)
    angles = np.arccos(np.clip(dot_prod, -1.0, 1.0))
    angles_deg = np.degrees(angles)

    return bool(np.any(angles_deg < min_angle_deg))


def generate_track(seed: int, point_count: int = 18, radius: float = 400.0,
                    track_width: float = 22.0, checkpoint_spacing: int = 8,
                    start_straight_length: float = 120.0) -> Track:
    """Генерує повну трасу за seed з гарантованою стартовою прямою. Використовує
    патерн 'Generate & Validate': якщо готова траса не відповідає вимогам (гострі
    кути, глобальне накладання асфальту, немає місця для старту), варіант
    відкидається і пробується наступний seed. Для зовнішнього коду результат все
    одно детермінований — той самий вхідний seed завжди повертає ту саму трасу."""
    current_seed = seed

    while True:
        rng = np.random.default_rng(current_seed)
        min_gap = track_width * 2.4

        points = _sample_points(rng, point_count, radius, min_gap=min_gap)
        edges = _build_delaunay_edges(points)
        mst_adj = _prim_mst(points, edges)
        cycle_indices = _tree_to_cycle(mst_adj, root=0)

        # 2-opt працює ТІЛЬКИ на графах (індексах точок), щоб розплутати структуру.
        cycle_indices = _remove_crossings(points, cycle_indices)

        control_points = points[cycle_indices]

        # Відкидаємо скелети з гострими "шпильками" (кут < 75°) ще до дорогої
        # математики сплайнів/офсетів — такий базовий кут завжди дасть затиснутий
        # поворот, незалежно від того, як сильно ми його потім згладжуватимемо.
        if _has_sharp_turns(control_points, min_angle_deg=75.0):
            current_seed += 1
            continue

        # Знаходимо найдовший відрізок базового скелета — кандидат на стартову пряму.
        diffs = np.roll(control_points, -1, axis=0) - control_points
        lengths = np.linalg.norm(diffs, axis=1)
        longest_idx = int(np.argmax(lengths))
        max_len = lengths[longest_idx]

        # Якщо найдовший відрізок закороткий для старту (із запасом 1.5) — інший seed.
        if max_len < start_straight_length * 1.5:
            current_seed += 1
            continue

        A = control_points[longest_idx]
        start_pos = A + 0.5 * diffs[longest_idx]
        start_dir = diffs[longest_idx] / max_len

        # Вставляємо точки строго на прямій між A і B — колінеарні точки Чайкін і
        # Catmull-Rom залишають ідеально рівними, тому тут гарантовано пряма ділянка.
        t = np.linspace(0, 1, 8)[1:-1]
        inserted_points = A + t[:, None] * diffs[longest_idx]
        control_points = np.insert(control_points, longest_idx + 1, inserted_points, axis=0)

        # Зрізаємо гострі кути (Чайкін) замість того, щоб тягнути точки до центру —
        # зберігає унікальну форму траси (не колапсує в коло), але кути стають досить
        # тупими, щоб офсетні нормалі (краї асфальту) не перекручувались.
        control_points = _chaikin_cut(control_points, iterations=4, ratio=0.25)

        # Завдяки Centripetal Catmull-Rom лінія більше не буде самоперетинатись,
        # тому другий прохід 2-opt на фізичних координатах тут не потрібен.
        center_line = _smooth_closed_loop(control_points, samples_per_segment=16)

        # Пост-обробка: розгладжуємо центральну лінію від мікро-вузлів.
        center_line = _smooth_dense_line(center_line, passes=3)

        if _is_track_valid(center_line, track_width):
            break
        current_seed += 1

    left_edge, right_edge = _offset_loop(center_line, track_width)

    # Пост-обробка: згладжуємо самі краї асфальту — прибирає "хвости ластівки"
    # (перекрути), які могли лишитись на офсетних контурах.
    left_edge = _smooth_dense_line(left_edge, passes=2)
    right_edge = _smooth_dense_line(right_edge, passes=2)

    checkpoints = center_line[::checkpoint_spacing]

    return Track(
        seed=seed,  # зберігаємо оригінальний вхідний seed для сумісності з іншим кодом
        center_line=center_line,
        left_edge=left_edge,
        right_edge=right_edge,
        checkpoints=checkpoints,
        control_points=control_points,
        start_pos=start_pos,
        start_dir=start_dir,
    )
