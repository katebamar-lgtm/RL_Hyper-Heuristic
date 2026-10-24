"""
MOGWO — Multi-Objective Grey Wolf Optimizer, d'après le schéma de mise à
jour de Mirjalili, Mirjalili & Lewis (2014, GWO mono-objectif) étendu au
multi-objectif façon Mirjalili, Saremi, Mirjalili & Coelho (2016, MOGWO).

Simplification documentée par rapport à l'article MOGWO original : la
sélection des leaders (alpha/beta/delta) dans l'archive externe s'y fait via
une grille adaptative hypercube (façon MOPSO de Coello & Lechuga) pour
répartir la pression de sélection. Ici, par souci d'homogénéité avec le
reste du projet (SPEA2/NSGA-II utilisent crowding distance, pas de grille),
les leaders sont tirés dans l'archive avec une probabilité proportionnelle
à leur distance de crowding (plus isolé = plus de chances d'être leader) —
un substitut fonctionnellement proche (favorise la diversité) mais plus
simple à maintenir cohérent avec `model/pareto.py`. À signaler explicitement
dans la section méthodologique de l'article.

Encodage/décodage : cf. `model/continuous_common.py` (vecteur de poids sur
les 4 objectifs, décodé par Dijkstra pondéré — partagé avec MOFA et MOABC).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Set

from .network import MultimodalNetwork
from .entities import Mode
from .pareto import crowding_distance, fast_non_dominated_sort
from .budget import EvalCounter
from .continuous_common import (
    Individual, N_OBJECTIVES, init_population, update_archive,
    evaluate_vector, random_weight_vector, normalize_weights,
)


@dataclass
class MOGWOParams:
    population_size: int = 40
    generations: int = 100
    archive_size: int = 40
    mutation_rate: float = 0.15  # cf. docstring module : palliatif à la
                                  # convergence prématurée du schéma GWO pur
    budget: Optional[int] = None
    seed: Optional[int] = None


def _pick_leaders(archive: List[Individual], rng: random.Random) -> List[List[float]]:
    """Tire 3 leaders (alpha/beta/delta) dans l'archive, pondérés par
    distance de crowding (cf. docstring module). Répétition autorisée si
    l'archive contient moins de 3 individus."""
    points = [ind.objectives for ind in archive]
    fronts = fast_non_dominated_sort(points)
    cd = crowding_distance(points, fronts[0]) if fronts else {}
    weights = [cd.get(i, 1.0) if cd.get(i, 1.0) != float('inf') else 10.0
               for i in range(len(archive))]
    total = sum(weights) or 1.0

    def _pick_one() -> List[float]:
        pick = rng.uniform(0, total)
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= pick:
                return archive[i].vector
        return archive[-1].vector

    return [_pick_one() for _ in range(3)]


def run_mogwo(net: MultimodalNetwork, origin: str, destination: str,
              allowed_modes: Set[Mode], params: Optional[MOGWOParams] = None,
              counter: Optional[EvalCounter] = None) -> List[Individual]:
    params = params or MOGWOParams()
    rng = random.Random(params.seed)
    counter = counter if counter is not None else EvalCounter(budget=params.budget)

    pop = init_population(net, origin, destination, allowed_modes,
                           params.population_size, rng, counter)
    if not pop:
        return []
    archive = update_archive(pop, params.archive_size)

    for gen in range(params.generations):
        if counter.exhausted() or not archive:
            break
        a = 2.0 - 2.0 * gen / max(1, params.generations - 1)  # décroît linéairement 2 -> 0

        new_pop: List[Individual] = []
        for wolf in pop:
            if counter.exhausted():
                new_pop.append(wolf)
                continue
            x_alpha, x_beta, x_delta = _pick_leaders(archive, rng)
            new_vector = []
            for j in range(N_OBJECTIVES):
                xs = []
                for leader in (x_alpha, x_beta, x_delta):
                    r1, r2 = rng.random(), rng.random()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    d = abs(C * leader[j] - wolf.vector[j])
                    xs.append(leader[j] - A * d)
                new_vector.append(sum(xs) / 3.0)
            if rng.random() < params.mutation_rate:
                new_vector = random_weight_vector(rng)
            new_vector = normalize_weights(new_vector)

            ind = evaluate_vector(net, new_vector, origin, destination, allowed_modes, counter)
            new_pop.append(ind if ind is not None else wolf)

        pop = new_pop
        archive = update_archive(pop + archive, params.archive_size)

    return archive
