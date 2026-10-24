"""
SPEA2 (Zitzler, Laumanns & Thiele, 2001) adapté à l'encodage "chemin".

Structure standard :
  - Fitness d'un individu i = R(i) + D(i)
      R(i) = somme des "forces" S(j) de tous les j (population + archive)
             qui dominent i ; S(j) = nombre d'individus que j domine.
      D(i) = 1 / (distance au k-ème plus proche voisin dans l'espace des
             objectifs + 2), k = sqrt(taille(pop) + taille(archive)).
  - Individus non dominés (R=0) -> fitness < 1.
  - Sélection environnementale : l'archive de la génération suivante est
    remplie par les non-dominés de (pop + archive) ; si trop nombreux on
    tronque en retirant itérativement l'individu le plus proche de son
    voisin ; si pas assez, on complète avec les meilleurs dominés restants.
  - Reproduction : tournoi binaire sur l'archive (fitness plus bas = mieux),
    croisement à nœud commun, mutation "re-routage standard" (pas OMF).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Set

from .network import MultimodalNetwork, Connection
from .entities import Mode
from .omf import random_walk_path
from .ga_common import shared_node_crossover, standard_mutation
from .pareto import dominates, normalize
from .budget import EvalCounter


@dataclass
class SPEA2Params:
    population_size: int = 60
    archive_size: int = 30
    generations: int = 100
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    max_hops: int = 40
    budget: Optional[int] = None  # cf. model/budget.py
    seed: Optional[int] = None


@dataclass
class Individual:
    path: List[Connection]
    objectives: tuple


def _evaluate(net: MultimodalNetwork, path: List[Connection],
              counter: Optional[EvalCounter] = None) -> tuple:
    if counter is not None:
        return counter.evaluate(net, path)
    return net.evaluate_route(path).as_tuple()


def _initial_population(net, origin, destination, allowed_modes, size, max_hops, rng,
                         counter: Optional[EvalCounter] = None) -> List[Individual]:
    pop: List[Individual] = []
    attempts = 0
    while len(pop) < size and attempts < size * 20:
        if counter is not None and counter.exhausted():
            break
        attempts += 1
        path = random_walk_path(net, origin, destination, allowed_modes,
                                 max_hops=max_hops, rng=rng)
        if path is not None:
            pop.append(Individual(path, _evaluate(net, path, counter)))
    return pop


def _compute_fitness(pool: List[Individual]) -> List[float]:
    n = len(pool)
    objs = [ind.objectives for ind in pool]

    # S(i) = nombre d'individus dominés par i ; R(i) = somme des S(j) qui dominent i
    dominates_matrix = [[dominates(objs[i], objs[j]) for j in range(n)] for i in range(n)]
    strength = [sum(dominates_matrix[i]) for i in range(n)]
    raw = [sum(strength[j] for j in range(n) if dominates_matrix[j][i]) for i in range(n)]

    # densité : distance au k-ème plus proche voisin, objectifs normalisés
    if n > 1:
        norm_objs, _, _ = normalize(objs)
        k = max(1, round(math.sqrt(n)))
        density = []
        for i in range(n):
            dists = sorted(math.dist(norm_objs[i], norm_objs[j]) for j in range(n) if j != i)
            sigma_k = dists[min(k - 1, len(dists) - 1)]
            density.append(1.0 / (sigma_k + 2.0))
    else:
        density = [0.0] * n

    return [raw[i] + density[i] for i in range(n)]


def _environmental_selection(pool: List[Individual], fitness: List[float],
                              archive_size: int) -> List[int]:
    non_dominated = [i for i in range(len(pool)) if fitness[i] < 1.0]

    if len(non_dominated) == archive_size:
        return non_dominated

    if len(non_dominated) < archive_size:
        # compléter avec les meilleurs dominés (fitness la plus basse)
        rest = sorted((i for i in range(len(pool)) if i not in non_dominated),
                       key=lambda i: fitness[i])
        return non_dominated + rest[:archive_size - len(non_dominated)]

    # trop de non-dominés : troncature par retrait itératif du plus proche voisin
    objs = [pool[i].objectives for i in non_dominated]
    norm_objs, _, _ = normalize(objs) if len(objs) > 1 else (objs, None, None)
    kept = list(range(len(non_dominated)))  # indices locaux à `non_dominated`
    while len(kept) > archive_size:
        # pour chaque candidat, sa distance au plus proche voisin parmi `kept`
        min_dists = []
        for a in kept:
            dists = sorted(math.dist(norm_objs[a], norm_objs[b]) for b in kept if b != a)
            min_dists.append((dists[0] if dists else float('inf'), a))
        min_dists.sort()
        kept.remove(min_dists[0][1])  # retire celui dont le plus proche voisin est le plus proche
    return [non_dominated[i] for i in kept]


def _tournament(archive: List[Individual], fitness: List[float], rng: random.Random) -> int:
    a, b = rng.sample(range(len(archive)), 2)
    return a if fitness[a] <= fitness[b] else b


def run_spea2(net: MultimodalNetwork, origin: str, destination: str,
              allowed_modes: Set[Mode], params: Optional[SPEA2Params] = None,
              counter: Optional[EvalCounter] = None
              ) -> List[Individual]:
    params = params or SPEA2Params()
    rng = random.Random(params.seed)
    counter = counter if counter is not None else EvalCounter(budget=params.budget)

    pop = _initial_population(net, origin, destination, allowed_modes,
                               params.population_size, params.max_hops, rng, counter)
    if not pop:
        return []
    archive: List[Individual] = []

    for _gen in range(params.generations):
        if counter.exhausted():
            break
        pool = pop + archive
        fitness = _compute_fitness(pool)
        selected_idx = _environmental_selection(pool, fitness, params.archive_size)
        archive = [pool[i] for i in selected_idx]
        archive_fitness = [fitness[i] for i in selected_idx]

        new_pop: List[Individual] = []
        while len(new_pop) < params.population_size and not counter.exhausted():
            i1 = _tournament(archive, archive_fitness, rng)
            i2 = _tournament(archive, archive_fitness, rng)
            p1, p2 = archive[i1].path, archive[i2].path

            if rng.random() < params.crossover_rate:
                child = shared_node_crossover(p1, p2, rng)
            else:
                child = rng.choice([p1, p2])
            if child is None:
                child = rng.choice([p1, p2])

            if rng.random() < params.mutation_rate:
                mutated = standard_mutation(net, child, allowed_modes, rng)
                if mutated is not None:
                    child = mutated

            new_pop.append(Individual(child, _evaluate(net, child, counter)))
        if not new_pop:
            break
        pop = new_pop

    pool = pop + archive
    fitness = _compute_fitness(pool)
    return [pool[i] for i in range(len(pool)) if fitness[i] < 1.0]
