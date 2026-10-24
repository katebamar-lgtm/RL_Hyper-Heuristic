"""
MOFA — Multi-Objective Firefly Algorithm, d'après le schéma d'attractivité
de Yang (2008/2009, FA mono-objectif) étendu au multi-objectif façon
Yang (2013, MOFA).

Adaptation multi-objectif : la notion de "luminosité" (inversement liée au
coût dans le FA mono-objectif) est remplacée par le comparateur de crowding
NSGA-II (rang de dominance puis distance de crowding) — même comparateur
que celui déjà utilisé dans les autres algorithmes du projet, pour rester homogène.
Une luciole i se déplace vers TOUTE luciole j "plus brillante" qu'elle
(rang inférieur, ou même rang et crowding supérieur) — fidèle au schéma
original où l'attraction est cumulative sur toutes les lucioles plus
brillantes, pas seulement la meilleure.

Encodage/décodage : cf. `model/continuous_common.py` (vecteur de poids sur
les 4 objectifs, décodé par Dijkstra pondéré — partagé avec MOGWO et MOABC).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Set

from .network import MultimodalNetwork
from .entities import Mode
from .pareto import fast_non_dominated_sort, crowding_distance
from .budget import EvalCounter
from .continuous_common import (
    Individual, N_OBJECTIVES, init_population, update_archive,
    evaluate_vector, normalize_weights,
)


@dataclass
class MOFAParams:
    population_size: int = 40
    generations: int = 100
    archive_size: int = 40
    beta0: float = 1.0      # attractivité à distance nulle
    gamma: float = 1.0      # coefficient d'absorption lumineuse
    alpha0: float = 0.3     # amplitude de randomisation initiale
    alpha_decay: float = 0.97  # décroissance géométrique de alpha par génération
    budget: Optional[int] = None
    seed: Optional[int] = None


def _rank_and_crowd(pop: List[Individual]):
    points = [ind.objectives for ind in pop]
    fronts = fast_non_dominated_sort(points)
    rank_of, crowd_of = {}, {}
    for rank, front in enumerate(fronts):
        cd = crowding_distance(points, front)
        for i in front:
            rank_of[i] = rank
            crowd_of[i] = cd[i]
    return rank_of, crowd_of


def _brighter(i: int, j: int, rank_of: dict, crowd_of: dict) -> bool:
    """j est-elle plus brillante que i ? (comparateur crowding NSGA-II)."""
    if rank_of[j] < rank_of[i]:
        return True
    if rank_of[j] == rank_of[i] and crowd_of[j] > crowd_of[i]:
        return True
    return False


def run_mofa(net: MultimodalNetwork, origin: str, destination: str,
              allowed_modes: Set[Mode], params: Optional[MOFAParams] = None,
              counter: Optional[EvalCounter] = None) -> List[Individual]:
    params = params or MOFAParams()
    rng = random.Random(params.seed)
    counter = counter if counter is not None else EvalCounter(budget=params.budget)

    pop = init_population(net, origin, destination, allowed_modes,
                           params.population_size, rng, counter)
    if not pop:
        return []
    archive = update_archive(pop, params.archive_size)

    alpha = params.alpha0
    for _gen in range(params.generations):
        if counter.exhausted():
            break
        rank_of, crowd_of = _rank_and_crowd(pop)
        n = len(pop)

        new_pop: List[Individual] = []
        for i in range(n):
            if counter.exhausted():
                new_pop.append(pop[i])
                continue
            brighter_idx = [j for j in range(n) if j != i and _brighter(i, j, rank_of, crowd_of)]

            new_vector = list(pop[i].vector)
            if brighter_idx:
                for j in brighter_idx:
                    r2 = sum((pop[i].vector[d] - pop[j].vector[d]) ** 2 for d in range(N_OBJECTIVES))
                    beta = params.beta0 * math.exp(-params.gamma * r2)
                    for d in range(N_OBJECTIVES):
                        new_vector[d] += beta * (pop[j].vector[d] - new_vector[d]) \
                                          + alpha * (rng.random() - 0.5)
            else:
                # personne de plus brillant (déjà non dominée) -> exploration locale
                new_vector = [v + alpha * (rng.random() - 0.5) for v in new_vector]
            new_vector = normalize_weights(new_vector)

            ind = evaluate_vector(net, new_vector, origin, destination, allowed_modes, counter)
            new_pop.append(ind if ind is not None else pop[i])

        pop = new_pop
        archive = update_archive(pop + archive, params.archive_size)
        alpha *= params.alpha_decay

    return archive
