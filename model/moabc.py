"""
MOABC — Multi-Objective Artificial Bee Colony, d'après le schéma ABC
mono-objectif de Karaboga (2005) étendu au multi-objectif par tri de
dominance et archive externe façon Zou, Wang, Hei, Zhu & Zhang (2011,
"Solving Multiobjective Optimization Problems Using Artificial Bee Colony
Algorithm", *Discrete Dynamics in Nature and Society*).

Écart documenté par rapport à des variantes plus élaborées de la
littérature (ex. Akbari et al. 2012, grille adaptative) : ici, comme pour
MOGWO/MOFA, la sélection des onlooker bees se fait par roulette pondérée
rang de dominance + crowding distance (comparateur NSGA-II déjà utilisé
partout ailleurs dans le projet), pas par une grille hypercube — choix de
cohérence méthodologique du projet, à signaler dans l'article.

Trois phases par génération (standard ABC) :
  1. Employed bees : chaque source de nourriture i génère un voisin en
     perturbant UNE dimension (cf. Karaboga 2005), accepté si non dominé
     par la source d'origine.
  2. Onlooker bees : autant de "abeilles suiveuses" que de sources ;
     chacune choisit une source par roulette (rang+crowding) puis répète
     la recherche de voisinage de l'étape 1.
  3. Scout bees : toute source non améliorée depuis `limit` tentatives
     consécutives est abandonnée et remplacée par une position aléatoire.

Encodage/décodage : cf. `model/continuous_common.py` (vecteur de poids sur
les 4 objectifs, décodé par Dijkstra pondéré — partagé avec MOGWO et MOFA).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Set

from .network import MultimodalNetwork
from .entities import Mode
from .pareto import dominates, fast_non_dominated_sort, crowding_distance
from .budget import EvalCounter
from .continuous_common import (
    Individual, N_OBJECTIVES, init_population, update_archive,
    evaluate_vector, random_weight_vector, normalize_weights,
)


@dataclass
class MOABCParams:
    population_size: int = 40   # nombre de sources de nourriture (SN)
    generations: int = 100
    limit: int = 10             # essais consécutifs sans amélioration avant scout
    archive_size: int = 40
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


def _roulette_pick(weights: List[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))
    pick = rng.uniform(0, total)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= pick:
            return i
    return len(weights) - 1


def _neighbor_vector(pop: List[Individual], i: int, rng: random.Random) -> List[float]:
    """Perturbation ABC standard : une seule dimension modifiée par
    combinaison avec une source voisine tirée au hasard (Karaboga 2005)."""
    others = [k for k in range(len(pop)) if k != i]
    k = rng.choice(others) if others else i
    d = rng.randrange(N_OBJECTIVES)
    phi = rng.uniform(-1.0, 1.0)
    new_vector = list(pop[i].vector)
    new_vector[d] = new_vector[d] + phi * (new_vector[d] - pop[k].vector[d])
    return normalize_weights(new_vector)


def run_moabc(net: MultimodalNetwork, origin: str, destination: str,
              allowed_modes: Set[Mode], params: Optional[MOABCParams] = None,
              counter: Optional[EvalCounter] = None) -> List[Individual]:
    params = params or MOABCParams()
    rng = random.Random(params.seed)
    counter = counter if counter is not None else EvalCounter(budget=params.budget)

    pop = init_population(net, origin, destination, allowed_modes,
                           params.population_size, rng, counter)
    if not pop:
        return []
    trials = [0] * len(pop)
    archive = update_archive(pop, params.archive_size)

    def _try_improve(i: int) -> None:
        new_vec = _neighbor_vector(pop, i, rng)
        cand = evaluate_vector(net, new_vec, origin, destination, allowed_modes, counter)
        # accepté si la source d'origine ne domine PAS le candidat (couvre
        # "candidat meilleur" et "mutuellement non dominés", cf. docstring)
        if cand is not None and not dominates(pop[i].objectives, cand.objectives):
            pop[i] = cand
            trials[i] = 0
        else:
            trials[i] += 1

    for _gen in range(params.generations):
        if counter.exhausted():
            break

        # 1) employed bees
        for i in range(len(pop)):
            if counter.exhausted():
                break
            _try_improve(i)

        # 2) onlooker bees (autant que de sources, sélection par roulette)
        rank_of, crowd_of = _rank_and_crowd(pop)
        max_rank = max(rank_of.values()) if rank_of else 0
        weights = []
        for idx in range(len(pop)):
            crowd = crowd_of.get(idx, 0.0)
            crowd = crowd if crowd != float('inf') else 1.0
            weights.append((max_rank - rank_of.get(idx, 0) + 1) + 0.01 * crowd)
        for _ in range(len(pop)):
            if counter.exhausted():
                break
            i = _roulette_pick(weights, rng)
            _try_improve(i)

        # 3) scout bees
        for i in range(len(pop)):
            if counter.exhausted():
                break
            if trials[i] >= params.limit:
                new_vec = random_weight_vector(rng)
                cand = evaluate_vector(net, new_vec, origin, destination, allowed_modes, counter)
                if cand is not None:
                    pop[i] = cand
                trials[i] = 0

        archive = update_archive(pop + archive, params.archive_size)

    return archive
