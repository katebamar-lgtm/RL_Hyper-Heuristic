"""
Opérateurs génériques partagés par les algorithmes de comparaison
(NSGA-II, SPEA2, MOEA/D).

IMPORTANT : ces opérateurs sont volontairement "standards" (littérature
générique), PAS basés sur OMF — sinon la comparaison vs
NSGA-II/SPEA2/MOEA/D ne serait pas équitable (on comparerait l'apport d'OMF
à... OMF).
"""

from __future__ import annotations
import random
from typing import List, Optional, Set

from .network import MultimodalNetwork, Connection
from .entities import Mode
from .omf import random_walk_path


def shared_node_crossover(p1: List[Connection], p2: List[Connection],
                           rng: random.Random) -> Optional[List[Connection]]:
    """Croisement à un point, adapté aux chemins de longueur variable :
    trouve un nœud commun aux deux parents (hors destination) et échange
    les suffixes à partir de ce nœud."""
    if not p1 or not p2:
        return None
    nodes1 = [c.destination for c in p1]
    nodes2 = {c.destination for c in p2}
    common = [n for n in nodes1[:-1] if n in nodes2]
    if not common:
        return None
    cut_node = rng.choice(common)
    idx1 = next(i for i, c in enumerate(p1) if c.destination == cut_node)
    idx2 = next(i for i, c in enumerate(p2) if c.destination == cut_node)
    return p1[:idx1 + 1] + p2[idx2 + 1:]


def standard_mutation(net: MultimodalNetwork, path: List[Connection],
                       allowed_modes: Set[Mode],
                       rng: random.Random) -> Optional[List[Connection]]:
    """Mutation générique "re-routage depuis un point aléatoire" : coupe le
    chemin à un nœud intermédiaire tiré au hasard, régénère la suite par
    marche aléatoire jusqu'à la destination. Opérateur standard de la
    littérature GA appliquée au routage — ne présuppose aucune des
    spécificités d'OMF (pas de distinction largeur/profondeur).
    """
    if len(path) < 2:
        return random_walk_path(net, path[0].origin, path[-1].destination,
                                 allowed_modes, rng=rng) if path else None
    cut_idx = rng.randrange(len(path) - 1)  # garde au moins 1 arc, évite de tout regénérer à chaque fois
    prefix = path[:cut_idx + 1]
    cut_node = prefix[-1].destination
    destination = path[-1].destination
    suffix = random_walk_path(net, cut_node, destination, allowed_modes,
                               max_hops=max(10, 2 * (len(path) - cut_idx)),
                               rng=rng, max_attempts=5)
    if suffix is None:
        return None
    return prefix + suffix


def binary_tournament(pop_indices: List[int], rank_of: dict, crowd_of: dict,
                       rng: random.Random) -> int:
    """Tournoi binaire NSGA-II : meilleur rang gagne, la distance de
    crowding départage les ex-aequo de rang."""
    a, b = rng.sample(pop_indices, 2)
    if rank_of[a] < rank_of[b]:
        return a
    if rank_of[b] < rank_of[a]:
        return b
    return a if crowd_of[a] >= crowd_of[b] else b
