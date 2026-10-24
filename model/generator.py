"""
Générateur de réseaux multimodaux aléatoires — portage Python de
webapp/core-logic.js (même logique, pour rester cohérent entre le
simulateur graphique et les expériences algorithmiques en Python).

Utilisé pour :
  - tester les algorithmes sur des réseaux plus grands que l'exemple à 8 nœuds ;
  - construire la suite d'instances de calibration et de benchmark final.
"""

from __future__ import annotations
import math
import random
from typing import List, Optional, Tuple

from .entities import Mode, TransitLine, LineSegment
from .network import MultimodalNetwork

MODE_PARAMS = {
    Mode.WALK: dict(speed=5.0, cost_per_km=0.0),
    Mode.BIKE: dict(speed=15.0, cost_per_km=0.15),
    Mode.CAR: dict(speed=40.0, cost_per_km=0.6),
}

LINE_PARAMS = {
    Mode.BUS:   dict(speed=22.0, fare=1.0, prefix="B", stop_ratio=(0.20, 0.30)),
    Mode.TRAM:  dict(speed=26.0, fare=1.5, prefix="T", stop_ratio=(0.30, 0.40)),
    Mode.METRO: dict(speed=34.0, fare=2.0, prefix="M", stop_ratio=(0.10, 0.20)),
}


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _generate_positions(n: int, w: float, h: float, min_sep: float,
                         rng: random.Random) -> List[Tuple[float, float]]:
    positions: List[Tuple[float, float]] = []
    for _ in range(n):
        p = (rng.uniform(0, w), rng.uniform(0, h))
        tries = 0
        while tries < 200 and any(_dist(p, q) < min_sep for q in positions):
            p = (rng.uniform(0, w), rng.uniform(0, h))
            tries += 1
        positions.append(p)
    return positions


def _mst_edges(positions: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    """Prim, retourne les arêtes (i, j) de l'arbre couvrant minimal."""
    n = len(positions)
    in_tree = [False] * n
    in_tree[0] = True
    edges = []
    count = 1
    while count < n:
        best = None
        for i in range(n):
            if not in_tree[i]:
                continue
            for j in range(n):
                if in_tree[j]:
                    continue
                d = _dist(positions[i], positions[j])
                if best is None or d < best[2]:
                    best = (i, j, d)
        in_tree[best[1]] = True
        edges.append((best[0], best[1]))
        count += 1
    return edges


def _knn_edges(positions: List[Tuple[float, float]], k: int) -> List[Tuple[int, int]]:
    n = len(positions)
    seen = set()
    edges = []
    for i in range(n):
        dists = sorted((j for j in range(n) if j != i),
                       key=lambda j: _dist(positions[i], positions[j]))
        for j in dists[:k]:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return edges


def _dedup(*edge_lists: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out = []
    for lst in edge_lists:
        for e in lst:
            key = (min(e), max(e))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _pick_line_stops(node_ids: List[str], positions: List[Tuple[float, float]],
                      count: int, rng: random.Random) -> List[str]:
    """Chaîne du plus proche voisin (avec un peu d'aléa) : parcours plausible
    d'une ligne, identique à webapp/core-logic.js::pickLineStops."""
    n = len(node_ids)
    remaining = list(range(n))
    start = remaining.pop(rng.randrange(len(remaining)))
    stops = [start]
    current = start
    while len(stops) < count and remaining:
        remaining.sort(key=lambda j: _dist(positions[current], positions[j]))
        pool = min(2, len(remaining))
        pick = rng.randrange(pool)
        nxt = remaining.pop(pick)
        stops.append(nxt)
        current = nxt
    return [node_ids[i] for i in stops]


def generate_network(n: int, nb: int, nt: int, nm: int,
                      w: float = 12.0, h: float = 9.0,
                      seed: Optional[int] = None) -> MultimodalNetwork:
    """Génère un réseau multimodal aléatoire.

    n  : nombre de nœuds
    nb, nt, nm : nombre de lignes de bus / tram / métro
    w, h : dimensions de la zone (km)
    seed : graine pour la reproductibilité (essentiel pour la calibration)
    """
    rng = random.Random(seed)
    min_sep = max(0.5, min(w, h) / math.sqrt(n) * 0.5)
    positions = _generate_positions(n, w, h, min_sep, rng)
    node_ids = [f"N{i+1}" for i in range(n)]

    net = MultimodalNetwork()
    for nid, (x, y) in zip(node_ids, positions):
        net.add_node(nid, x, y)

    mst_e = _mst_edges(positions)
    knn_e = _knn_edges(positions, k=3)
    street_e = _dedup(mst_e, knn_e)

    eid = 0
    for mode, edge_set in ((Mode.WALK, street_e), (Mode.BIKE, street_e), (Mode.CAR, mst_e)):
        params = MODE_PARAMS[mode]
        for i, j in edge_set:
            d = _dist(positions[i], positions[j])
            time = (d / params["speed"]) * 60.0
            cost = round(d * params["cost_per_km"], 2)
            net.add_edge(f"E{eid}", node_ids[i], node_ids[j], mode,
                         distance=d, time=time, cost=cost, bidirectional=True)
            eid += 1

    for mode, count in ((Mode.BUS, nb), (Mode.TRAM, nt), (Mode.METRO, nm)):
        params = LINE_PARAMS[mode]
        lo, hi = params["stop_ratio"]
        for i in range(1, count + 1):
            ratio = rng.uniform(lo, hi)
            stop_count = max(2, min(n, round(n * ratio)))
            stop_ids = _pick_line_stops(node_ids, positions, stop_count, rng)
            segments = []
            for s in range(len(stop_ids) - 1):
                a = net.nodes[stop_ids[s]].coord
                b = net.nodes[stop_ids[s + 1]].coord
                d = _dist((a.x, a.y), (b.x, b.y))
                segments.append(LineSegment(stop_ids[s], stop_ids[s + 1], d,
                                             (d / params["speed"]) * 60.0, cost=0.0))
            line = TransitLine(f"{params['prefix']}{i}", f"{mode.value.title()} {i}",
                               mode, stop_ids, segments, fare=params["fare"])
            net.add_line(line)

    net.build_graph()
    return net
