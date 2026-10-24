"""
MOEA/D (Zhang & Li, 2007) — décomposition en sous-problèmes scalaires,
adapté à l'encodage "chemin".

Structure standard :
  - N vecteurs de poids w_1..w_N sur le simplexe (4 objectifs -> tirage
    Dirichlet uniforme, alternative pratique au treillis de Das-Dennis
    quand la dimension ne se prête pas à un treillis exact avec N fixé).
  - Voisinage B(i) = les T vecteurs de poids les plus proches de w_i
    (distance euclidienne dans l'espace des poids).
  - Point idéal z* = minimum courant par objectif sur toute la population.
  - Scalarisation de Tchebycheff : g(x|w,z*) = max_k w_k * |f_k(x) - z*_k|.
  - Boucle : pour chaque sous-problème i, parents tirés dans B(i),
    croisement + mutation standard (pas OMF), remplacement des voisins de
    B(i) si le nouvel individu améliore leur g(.), limité à `nr` remplacements
    par génération pour éviter qu'un sous-problème n'envahisse tout le
    voisinage.
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
from .pareto import non_dominated_indices
from .budget import EvalCounter

N_OBJECTIVES = 4  # temps, distance, coût, correspondances


@dataclass
class MOEADParams:
    population_size: int = 60   # = nombre de sous-problèmes N
    generations: int = 100
    neighborhood_size: int = 8  # T
    max_replacements: int = 2   # nr
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


def _generate_weight_vectors(n: int, m: int, rng: random.Random) -> List[List[float]]:
    """Tirage uniforme sur le simplexe (m dimensions) par normalisation de
    variables exponentielles i.i.d. — équivalent à Dirichlet(1,...,1)."""
    vectors = []
    for _ in range(n):
        exps = [-math.log(rng.random() + 1e-12) for _ in range(m)]
        s = sum(exps)
        vectors.append([e / s for e in exps])
    return vectors


def _neighborhoods(weights: List[List[float]], t: int) -> List[List[int]]:
    n = len(weights)
    neigh = []
    for i in range(n):
        dists = sorted(range(n), key=lambda j: math.dist(weights[i], weights[j]))
        neigh.append(dists[:t])
    return neigh


def _tchebycheff(objectives: tuple, weight: List[float], ideal: List[float]) -> float:
    return max(w * abs(f - z) for w, f, z in zip(weight, objectives, ideal))


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


def run_moead(net: MultimodalNetwork, origin: str, destination: str,
              allowed_modes: Set[Mode], params: Optional[MOEADParams] = None,
              counter: Optional[EvalCounter] = None
              ) -> List[Individual]:
    params = params or MOEADParams()
    rng = random.Random(params.seed)
    counter = counter if counter is not None else EvalCounter(budget=params.budget)

    pop = _initial_population(net, origin, destination, allowed_modes,
                               params.population_size, params.max_hops, rng, counter)
    if not pop:
        return []
    n = len(pop)

    weights = _generate_weight_vectors(n, N_OBJECTIVES, rng)
    neighborhoods = _neighborhoods(weights, min(params.neighborhood_size, n))
    ideal = [min(ind.objectives[k] for ind in pop) for k in range(N_OBJECTIVES)]

    for _gen in range(params.generations):
        if counter.exhausted():
            break
        for i in range(n):
            if counter.exhausted():
                break
            b_i = neighborhoods[i]
            j1, j2 = rng.sample(b_i, 2) if len(b_i) >= 2 else (i, i)
            p1, p2 = pop[j1].path, pop[j2].path

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

            child_obj = _evaluate(net, child, counter)
            ideal = [min(ideal[k], child_obj[k]) for k in range(N_OBJECTIVES)]

            replacements = 0
            order = list(b_i)
            rng.shuffle(order)
            for j in order:
                if replacements >= params.max_replacements:
                    break
                g_child = _tchebycheff(child_obj, weights[j], ideal)
                g_current = _tchebycheff(pop[j].objectives, weights[j], ideal)
                if g_child <= g_current:
                    pop[j] = Individual(child, child_obj)
                    replacements += 1

    objs = [ind.objectives for ind in pop]
    nd_idx = non_dominated_indices(objs)
    return [pop[i] for i in nd_idx]
