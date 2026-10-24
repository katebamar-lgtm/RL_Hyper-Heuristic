"""
Utilitaires Pareto ET indicateurs de performance multi-objectif, communs à
tous les algorithmes du projet (NSGA-II, SPEA2, MOEA/D, MOGWO,
MOFA, MOABC à venir) et à l'évaluation comparative finale.

Convention : un "point objectif" est un tuple/liste de floats, TOUJOURS À
MINIMISER (cf. RouteObjectives.as_tuple() dans network.py).

Historique : ce module fusionne l'ancien `pareto.py` (dominance, tri non
dominé, crowding, hypervolume — utilisé par les algorithmes) et l'ancien
`indicators.py` (jamais importé nulle part dans le projet, contenait du
code mort dans `fast_non_dominated_sort` et une définition de dominance
redondante/incohérente avec celle-ci). Fusion en un seul module pour éviter
que deux implémentations divergentes de la dominance coexistent (risque de
résultats incohérents selon le module réellement utilisé par un script).
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """a domine b au sens large (Def. 2, Dib et al. 2023) : a <= b sur tous
    les critères, et strictement < sur au moins un."""
    not_worse = True
    strictly_better = False
    for x, y in zip(a, b):
        if x > y:
            not_worse = False
            break
        if x < y:
            strictly_better = True
    return not_worse and strictly_better


def strictly_dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """a domine strictement b (Def. 3) : meilleur sur TOUS les critères."""
    return all(x < y for x, y in zip(a, b))


def non_dominated_indices(points: List[Sequence[float]]) -> List[int]:
    """Indices des points non dominés (front de rang 1), O(n^2) — suffisant
    pour les tailles de population de ce projet."""
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if dominates(points[j], points[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def fast_non_dominated_sort(points: List[Sequence[float]]) -> List[List[int]]:
    """Tri non dominé façon NSGA-II (Deb et al. 2002) : liste de fronts
    (listes d'indices), fronts[0] = rang 1 (meilleur)."""
    n = len(points)
    domination_count = [0] * n
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(points[i], points[j]):
                dominated_by[i].append(j)
            elif dominates(points[j], points[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def crowding_distance(points: List[Sequence[float]], front: List[int]) -> Dict[int, float]:
    """Distance de crowding (NSGA-II) pour les indices d'un front donné."""
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = float('inf')
        return distance
    m = len(points[0])
    for obj_idx in range(m):
        front_sorted = sorted(front, key=lambda i: points[i][obj_idx])
        distance[front_sorted[0]] = float('inf')
        distance[front_sorted[-1]] = float('inf')
        lo = points[front_sorted[0]][obj_idx]
        hi = points[front_sorted[-1]][obj_idx]
        span = hi - lo
        if span == 0:
            continue
        for k in range(1, len(front_sorted) - 1):
            prev_v = points[front_sorted[k - 1]][obj_idx]
            next_v = points[front_sorted[k + 1]][obj_idx]
            distance[front_sorted[k]] += (next_v - prev_v) / span
    return distance


def normalize(points: List[Sequence[float]], ideal=None, nadir=None):
    """Normalisation min-max colonne par colonne dans [0,1]."""
    m = len(points[0])
    if ideal is None:
        ideal = [min(p[k] for p in points) for k in range(m)]
    if nadir is None:
        nadir = [max(p[k] for p in points) for k in range(m)]
    out = []
    for p in points:
        row = []
        for k in range(m):
            span = nadir[k] - ideal[k]
            row.append(0.0 if span == 0 else (p[k] - ideal[k]) / span)
        out.append(tuple(row))
    return out, ideal, nadir


def elitist_truncate(points: List[Sequence[float]], max_size: int) -> List[int]:
    """Sélection élitiste générique façon NSGA-II : remplit par fronts de
    dominance croissants, tronque le dernier front admis par distance de
    crowding décroissante. Retourne les INDICES (dans `points`) à conserver.

    Factorisé ici pour être réutilisé par MOGWO/MOFA/MOABC (gestion de
    leur archive externe) sans dupliquer la logique déjà écrite trois fois
    dans les algorithmes génétiques du projet (remplacement générationnel)."""
    fronts = fast_non_dominated_sort(points)
    kept: List[int] = []
    for front in fronts:
        if len(kept) + len(front) <= max_size:
            kept.extend(front)
        else:
            remaining = max_size - len(kept)
            if remaining > 0:
                cd = crowding_distance(points, front)
                front_sorted = sorted(front, key=lambda i: -cd[i])
                kept.extend(front_sorted[:remaining])
            break
    return kept


def hypervolume_2d(points: List[Sequence[float]], reference: Sequence[float]) -> float:
    """Hypervolume EXACT pour 2 objectifs à minimiser (balayage), utile pour
    valider l'estimateur Monte-Carlo ci-dessous. `points` doit être filtré
    aux non-dominés au préalable."""
    pts_sorted = sorted(points, key=lambda p: p[0])
    hv = 0.0
    last_y = reference[1]
    for x, y in pts_sorted:
        if y < last_y:
            hv += (reference[0] - x) * (last_y - y)
            last_y = y
    return hv


def hypervolume(points: List[Sequence[float]], reference: Sequence[float],
                 ideal: Sequence[float], n_samples: int = 20000,
                 rng: random.Random = None) -> float:
    """Estimation Monte-Carlo de l'hypervolume, pour un nombre d'objectifs
    quelconque (calcul exact = #P-difficile au-delà de 3 objectifs ; MC est
    la pratique standard en optimisation many-objective). Recommandé sur des
    objectifs déjà normalisés (ideal=(0,...,0), reference=(1,...,1) + marge).
    """
    rng = rng or random.Random()
    m = len(reference)
    non_dom = [points[i] for i in non_dominated_indices(points)]
    if not non_dom:
        return 0.0
    count = 0
    for _ in range(n_samples):
        sample = [rng.uniform(ideal[k], reference[k]) for k in range(m)]
        for p in non_dom:
            if all(p[k] <= sample[k] for k in range(m)):
                count += 1
                break
    box_volume = 1.0
    for k in range(m):
        box_volume *= (reference[k] - ideal[k])
    return (count / n_samples) * box_volume


# ------------------------------------------------------------------ #
# Bornes idéal/nadir (utilisé par hv_rpd et par IGD/GD ci-dessous)
# ------------------------------------------------------------------ #

def bounds(points: List[Sequence[float]]) -> Tuple[tuple, tuple]:
    """Point idéal (min par objectif) et point nadir (max par objectif)."""
    m = len(points[0])
    ideal = tuple(min(p[k] for p in points) for k in range(m))
    nadir = tuple(max(p[k] for p in points) for k in range(m))
    return ideal, nadir


# ------------------------------------------------------------------ #
# IGD / GD — attendus par la plupart des reviewers en optimisation
# multi-objectif en complément de l'hypervolume (ex. Dib 2023, Scientific
# Reports, qui rapporte HV + coverage + epsilon + GD + IGD + spread).
#
# Nécessitent un front de RÉFÉRENCE : en l'absence du vrai front de Pareto
# (NP-difficile à obtenir exactement ici), la pratique standard consiste à
# utiliser l'union des fronts non dominés obtenus par TOUS les algorithmes
# comparés sur une instance donnée, filtrée aux non-dominés — cf. Phase 4
# du protocole expérimental.
# ------------------------------------------------------------------ #

def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _normalize_pair(candidate: Sequence[Sequence[float]],
                     reference: Sequence[Sequence[float]]
                     ) -> Tuple[List[tuple], List[tuple]]:
    """Normalise candidate et reference ENSEMBLE (mêmes bornes idéal/nadir
    tirées de leur union), condition nécessaire pour que les distances
    inter-objectifs (temps en minutes, coût en unité monétaire, nombre de
    correspondances entier...) restent comparables."""
    union = list(candidate) + list(reference)
    ideal, nadir = bounds(union)
    cand_norm, _, _ = normalize(list(candidate), ideal, nadir)
    ref_norm, _, _ = normalize(list(reference), ideal, nadir)
    return cand_norm, ref_norm


def generational_distance(candidate: Sequence[Sequence[float]],
                           reference: Sequence[Sequence[float]]) -> float:
    """GD(candidate) : distance moyenne de chaque point candidat à son plus
    proche voisin dans le front de référence (objectifs normalisés
    ensemble). 0 = tous les points candidats sont sur la référence.
    Mesure la PROXIMITÉ mais pas la couverture (un candidat réduit à un
    seul point proche de la référence peut avoir un GD excellent tout en
    étant un très mauvais front) — toujours reporter IGD à côté."""
    if not candidate:
        return float('inf')
    if not reference:
        raise ValueError("le front de référence est vide")
    cand_norm, ref_norm = _normalize_pair(candidate, reference)
    dists = [min(_euclidean(c, r) for r in ref_norm) for c in cand_norm]
    return sum(dists) / len(dists)


def inverted_generational_distance(candidate: Sequence[Sequence[float]],
                                    reference: Sequence[Sequence[float]]) -> float:
    """IGD(candidate) : distance moyenne de chaque point de RÉFÉRENCE à son
    plus proche voisin dans le front candidat (objectifs normalisés
    ensemble). 0 = le candidat couvre parfaitement la référence. Contraint
    à la fois la proximité ET la diversité/couverture — c'est l'indicateur
    le plus systématiquement demandé par les reviewers en complément de
    l'hypervolume (avec GD, moins discriminant seul, en complément
    secondaire)."""
    if not reference:
        raise ValueError("le front de référence est vide")
    if not candidate:
        return float('inf')  # aucune solution -> distance infinie à la référence
    cand_norm, ref_norm = _normalize_pair(candidate, reference)
    dists = [min(_euclidean(r, c) for c in cand_norm) for r in ref_norm]
    return sum(dists) / len(dists)


def hv_rpd(candidate: Sequence[Sequence[float]], reference: Sequence[Sequence[float]],
           ref_point: Optional[Sequence[float]] = None, n_samples: int = 20000,
           rng: Optional[random.Random] = None) -> float:
    """I_H^rpd(candidate) = (H(reference) - H(candidate)) / H(reference).

    Complémentaire de l'hypervolume brut : exprime l'écart RELATIF entre le
    candidat et une référence, ce qui facilite l'agrégation entre instances
    de tailles différentes (contrairement à l'hypervolume brut, dont
    l'échelle dépend de l'instance). 0 = candidat atteint le même
    hypervolume que la référence. Proche de 1 = très en retrait. Peut être
    négatif si candidate fait mieux que reference (front de référence non
    exact, ex. approximation par union plutôt que vrai front).

    Repris de l'ancien `indicators.py` (protocole de marge +10% au-delà du
    nadir normalisé, cf. Bezerra et al. 2016), adapté aux primitives de ce
    module (`normalize`, `hypervolume`).
    """
    if not reference:
        raise ValueError("le front de référence est vide")
    if not candidate:
        return 1.0  # aucune solution -> pire cas possible

    cand_norm, ref_norm = _normalize_pair(candidate, reference)
    m = len(cand_norm[0]) if cand_norm else len(ref_norm[0])
    rp = tuple(ref_point) if ref_point is not None else tuple([1.1] * m)
    ideal = tuple([0.0] * m)

    rng = rng or random.Random()
    h_ref = hypervolume(list(ref_norm), rp, ideal, n_samples=n_samples, rng=rng)
    h_cand = hypervolume(list(cand_norm), rp, ideal, n_samples=n_samples, rng=rng)
    if h_ref <= 1e-12:
        return 0.0
    return (h_ref - h_cand) / h_ref
