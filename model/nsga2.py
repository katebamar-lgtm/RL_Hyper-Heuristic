"""
NSGA-II (Deb et al., 2002) adapté à l'encodage "chemin dans un réseau
multimodal" — utilisé comme référence de comparaison.

Structure standard :
  1. Population initiale (RPG = marche aléatoire).
  2. Boucle : sélection par tournoi binaire (rang + crowding), croisement à
     nœud commun, mutation "re-routage standard" (PAS OMF), évaluation,
     remplacement générationnel élitiste (tri non dominé + troncature par
     crowding sur la population combinée parents+enfants).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Set

from .network import MultimodalNetwork, Connection
from .entities import Mode
from .omf import random_walk_path
from .ga_common import shared_node_crossover, standard_mutation, binary_tournament
from .pareto import fast_non_dominated_sort, crowding_distance
from .budget import EvalCounter


@dataclass
class NSGA2Params:
    population_size: int = 60
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


def _initial_population(net, origin, destination, allowed_modes, params, rng,
                         counter: Optional[EvalCounter] = None) -> List[Individual]:
    pop: List[Individual] = []
    attempts = 0
    max_attempts = params.population_size * 20
    while len(pop) < params.population_size and attempts < max_attempts:
        if counter is not None and counter.exhausted():
            break
        attempts += 1
        path = random_walk_path(net, origin, destination, allowed_modes,
                                 max_hops=params.max_hops, rng=rng)
        if path is not None:
            pop.append(Individual(path, _evaluate(net, path, counter)))
    return pop


def _rank_and_crowding(pop: List[Individual]):
    points = [ind.objectives for ind in pop]
    fronts = fast_non_dominated_sort(points)
    rank_of, crowd_of = {}, {}
    for rank, front in enumerate(fronts):
        cd = crowding_distance(points, front)
        for i in front:
            rank_of[i] = rank
            crowd_of[i] = cd[i]
    return fronts, rank_of, crowd_of


def run_nsga2(net: MultimodalNetwork, origin: str, destination: str,
              allowed_modes: Set[Mode], params: Optional[NSGA2Params] = None,
              counter: Optional[EvalCounter] = None
              ) -> List[Individual]:
    params = params or NSGA2Params()
    rng = random.Random(params.seed)
    counter = counter if counter is not None else EvalCounter(budget=params.budget)

    pop = _initial_population(net, origin, destination, allowed_modes, params, rng, counter)
    if not pop:
        return []

    for _gen in range(params.generations):
        if counter.exhausted():
            break
        _, rank_of, crowd_of = _rank_and_crowding(pop)
        indices = list(range(len(pop)))

        offspring: List[Individual] = []
        while len(offspring) < params.population_size and not counter.exhausted():
            i1 = binary_tournament(indices, rank_of, crowd_of, rng)
            i2 = binary_tournament(indices, rank_of, crowd_of, rng)
            p1, p2 = pop[i1].path, pop[i2].path

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

            offspring.append(Individual(child, _evaluate(net, child, counter)))

        combined = pop + offspring
        fronts, _, crowd_of = _rank_and_crowding(combined)
        new_pop: List[Individual] = []
        for front in fronts:
            if len(new_pop) + len(front) <= params.population_size:
                new_pop.extend(combined[i] for i in front)
            else:
                remaining = params.population_size - len(new_pop)
                front_sorted = sorted(front, key=lambda i: -crowd_of[i])
                new_pop.extend(combined[i] for i in front_sorted[:remaining])
                break
        pop = new_pop

    fronts, _, _ = _rank_and_crowding(pop)
    return [pop[i] for i in fronts[0]] if fronts else []
