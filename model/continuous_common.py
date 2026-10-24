"""
Briques communes à MOGWO, MOFA et MOABC (Phase 1) : ces trois métaheuristiques
sont nativement CONTINUES (elles manipulent des vecteurs de réels), alors que
le problème est un plus court chemin dans un graphe. Pour les comparer
équitablement entre elles (mêmes conditions), on fige UNE fois pour toutes un
schéma d'encodage/décodage commun, réutilisé à l'identique par les trois.

Encodage retenu : vecteur de POIDS sur les 4 objectifs (temps, distance,
coût, correspondances), un point du simplexe de dimension 4. Décodage :
Dijkstra à état minimisant la somme pondérée (`scenario.weighted_shortest_path`,
réutilise exactement la logique déjà validée de `evaluate_route`).

Historique de conception : un premier essai encodait un vecteur de PRIORITÉS
PAR NŒUD (dimension |V|), décodé par une marche gloutonne priorité-guidée.
Abandonné après diagnostic empirique : même par pur échantillonnage aléatoire
(sans aucune recherche adaptative), cet encodage ne trouvait qu'un seul point
de qualité sur le réseau de test (35 non-dominés pour un tirage RPG classique
contre 14, quasi tous dupliqués sur le même point, pour l'encodage par
priorités) — la marche gloutonne biaise structurellement vers les arcs de
marche (WALK), plus nombreux à chaque nœud que les correspondances de
transport en commun, empêchant de chaîner plusieurs tronçons TC. L'encodage
par vecteur de poids + Dijkstra pondéré n'a pas ce biais : chaque poids
donne un chemin optimal pour SA combinaison d'objectifs, quel que soit le
mode. Limite à documenter dans l'article : une scalarisation par somme
pondérée n'atteint que l'enveloppe convexe du front de Pareto (cf. docstring
de `weighted_shortest_path`).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from .network import MultimodalNetwork, Connection
from .entities import Mode
from .scenario import weighted_shortest_path
from .pareto import fast_non_dominated_sort, crowding_distance
from .budget import EvalCounter

N_OBJECTIVES = 4  # temps, distance, coût, correspondances — dimension du vecteur


@dataclass
class Individual:
    path: List[Connection]
    objectives: tuple
    vector: List[float]  # vecteur de poids ayant produit ce chemin (pour les
                          # opérateurs de mise à jour GWO/FA/ABC)


def random_weight_vector(rng: random.Random) -> List[float]:
    """Tirage uniforme sur le simplexe de dimension N_OBJECTIVES, par
    normalisation de variables exponentielles i.i.d. — même technique que
    `model.moead._generate_weight_vectors`, réutilisée ici pour cohérence."""
    exps = [-math.log(rng.random() + 1e-12) for _ in range(N_OBJECTIVES)]
    s = sum(exps)
    return [e / s for e in exps]


def normalize_weights(v: Sequence[float]) -> List[float]:
    """Reprojette un vecteur quelconque (issu d'une mise à jour GWO/FA/ABC,
    potentiellement négatif ou hors simplexe) sur le simplexe positif : on
    prend la valeur absolue (évite un poids négatif, qui casserait la
    non-négativité requise par Dijkstra) puis on normalise à somme 1. Un
    plancher (1e-9) évite un vecteur nul dégénéré."""
    absv = [abs(x) + 1e-9 for x in v]
    s = sum(absv)
    return [x / s for x in absv]


def decode_weight_vector(net: MultimodalNetwork, vector: Sequence[float],
                          origin: str, destination: str,
                          allowed_modes: Set[Mode]) -> Optional[List[Connection]]:
    return weighted_shortest_path(net, origin, destination, tuple(vector), allowed_modes)


def evaluate_vector(net: MultimodalNetwork, vector: Sequence[float],
                     origin: str, destination: str, allowed_modes: Set[Mode],
                     counter: Optional[EvalCounter] = None) -> Optional[Individual]:
    """Décode + évalue un vecteur de poids ; None si aucun chemin n'existe
    topologiquement entre origin et destination (indépendant du vecteur)."""
    path = decode_weight_vector(net, vector, origin, destination, allowed_modes)
    if path is None:
        return None
    obj = counter.evaluate(net, path) if counter is not None else net.evaluate_route(path).as_tuple()
    return Individual(path, obj, list(vector))


def init_population(net: MultimodalNetwork, origin: str, destination: str,
                     allowed_modes: Set[Mode], size: int, rng: random.Random,
                     counter: Optional[EvalCounter] = None) -> List[Individual]:
    pop: List[Individual] = []
    attempts = 0
    max_attempts = size * 5  # le décodage réussit quasi toujours (Dijkstra),
                              # peu de tentatives nécessaires contrairement
                              # à l'ancien décodage par marche aléatoire
    while len(pop) < size and attempts < max_attempts:
        if counter is not None and counter.exhausted():
            break
        attempts += 1
        vec = random_weight_vector(rng)
        ind = evaluate_vector(net, vec, origin, destination, allowed_modes, counter)
        if ind is not None:
            pop.append(ind)
        else:
            break  # pas de chemin topologique -> inutile de réessayer
    return pop


def update_archive(pool: List[Individual], archive_size: int) -> List[Individual]:
    """Archive externe générique (non-dominés uniquement + troncature par
    crowding distance si trop nombreux) — réutilisée à l'identique par
    MOGWO, MOFA et MOABC (Phase 1), pour que le mécanisme d'élitisme ne
    soit pas une variable de confusion entre les trois lors de la
    comparaison finale.

    L'archive n'est PAS complétée avec des individus dominés quand le
    front 0 est plus petit que `archive_size` (l'archive peut alors être
    plus petite que la cible, ce qui est normal et attendu pour une
    archive de Pareto externe) — ce sont ces archives qui sont directement
    renvoyées comme front final par `run_mogwo`/`run_mofa`/`run_moabc`.

    Dédoublonnage PAR OBJECTIFS avant tri : plusieurs vecteurs de poids
    différents peuvent décoder vers le même chemin optimal (plusieurs
    combinaisons de poids ont le même argmin) ; deux points objectifs
    identiques se déclarent mutuellement "non dominés", ce qui, sans
    dédoublonnage, peut fausser la troncature par crowding distance (les
    doublons se comportent comme des points frontière avec une distance de
    crowding artificiellement élevée). Un seul représentant par objectif
    distinct est conservé.
    """
    if not pool:
        return []
    seen = set()
    dedup: List[Individual] = []
    for ind in pool:
        if ind.objectives not in seen:
            seen.add(ind.objectives)
            dedup.append(ind)

    points = [ind.objectives for ind in dedup]
    fronts = fast_non_dominated_sort(points)
    if not fronts:
        return []
    first_front = fronts[0]
    if len(first_front) <= archive_size:
        return [dedup[i] for i in first_front]
    cd = crowding_distance(points, first_front)
    sorted_front = sorted(first_front, key=lambda i: -cd[i])
    return [dedup[i] for i in sorted_front[:archive_size]]
