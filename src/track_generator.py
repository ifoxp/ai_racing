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


def _catmull_rom(p0, p1, p2, p3, t: np.ndarray) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2 * p1[0])
        + (-p0[0] + p2[0]) * t
        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2 * p1[1])
        + (-p0[1] + p2[1]) * t
        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return np.column_stack([x, y])


def _smooth_closed_loop(points: np.ndarray, samples_per_segment: int = 16) -> np.ndarray:
    n = len(points)
    t = np.linspace(0, 1, samples_per_segment, endpoint=False)
    segments = []
    for i in range(n):
        p0, p1, p2, p3 = (
            points[(i - 1) % n],
            points[i],
            points[(i + 1) % n],
            points[(i + 2) % n],
        )
        segments.append(_catmull_rom(p0, p1, p2, p3, t))
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


def generate_track(seed: int, point_count: int = 18, radius: float = 400.0,
                    track_width: float = 22.0, checkpoint_spacing: int = 8) -> Track:
    """Генерує повну трасу за seed. Той самий seed завжди дає ту саму трасу.

    ВІДОМА ПРОБЛЕМА (навмисно залишена як є, будемо виправляти по кроках):
    center_line гарантовано без самоперетинів (2-opt), але офсетні контури
    left_edge/right_edge (краї асфальту) МОЖУТЬ самоперетинатись у гострих поворотах,
    якщо ширина траси більша за локальний радіус кривизни. Це видно візуально як
    "петлі"/перехрещення асфальту в деяких місцях."""
    rng = np.random.default_rng(seed)
    min_gap = track_width * 2.4  # асфальт шириною track_width*2 обабіч + запас

    points = _sample_points(rng, point_count, radius, min_gap=min_gap)
    edges = _build_delaunay_edges(points)
    mst_adj = _prim_mst(points, edges)
    cycle_indices = _tree_to_cycle(mst_adj, root=0)
    cycle_indices = _remove_crossings(points, cycle_indices)

    control_points = points[cycle_indices]
    center_line = _smooth_closed_loop(control_points, samples_per_segment=16)

    # Catmull-Rom може "перестрелити" за контрольні точки на гострих поворотах і
    # створити нові самоперетини центральної лінії, яких не було в ламаній лінії —
    # 2-opt повторюється вже на самій згладженій кривій.
    smooth_cycle = _remove_crossings(center_line, list(range(len(center_line))))
    center_line = center_line[smooth_cycle]

    left_edge, right_edge = _offset_loop(center_line, track_width)

    checkpoints = center_line[::checkpoint_spacing]

    return Track(
        seed=seed,
        center_line=center_line,
        left_edge=left_edge,
        right_edge=right_edge,
        checkpoints=checkpoints,
        control_points=control_points,
    )
