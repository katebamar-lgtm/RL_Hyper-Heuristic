"""
OMF — Optimisation par Filtres Morphologiques.

Recherche locale multicritère sur des chemins d'un réseau multimodal,
combinant deux structures de voisinage (d'après les figures fournies) :

  - "largeur"    : régénère un chemin topologiquement DIFFÉRENT entre la
                    même origine et la même destination (Fig. 3).
  - "profondeur" : garde EXACTEMENT la même séquence de nœuds, change le
                    mode de transport utilisé sur un arc du chemin (Fig. 4).

Paramètres (au sens du schéma 1) :
  - NN : nombre de voisins générés par itération
  - IT : nombre d'itérations
  - R  : seuil de choix entre les deux voisinages. À chaque itération, on
         tire rv ~ Uniform(0,1) ; si R < rv on utilise le voisinage en
         largeur, sinon le voisinage en profondeur (repris tel quel du
         schéma : "R < Random value in [0,1] ? Yes -> largeur, No -> profondeur").
         Donc P(largeur) = 1 - R, P(profondeur) = R.

Ces trois paramètres (+ le type de dominance) sont ceux que la calibration
(hypervolume, budget fixe) devra régler.
"""

from __future__ import annotations
import random
from typing import List, Optional, Sequence, Set

from .network import MultimodalNetwork, Connection
from .entities import Mode
from .pareto import dominates, strictly_dominates
from .budget import EvalCounter


def random_walk_path(net: MultimodalNetwork, origin: str, destination: str,
                      allowed_modes: Set[Mode], max_hops: int = 40,
                      rng: Optional[random.Random] = None,
                      max_attempts: int = 30) -> Optional[List[Connection]]:
    """Génère un chemin par marche aléatoire non répétitive entre origine et
    destination. Sert à la fois de RPG (population initiale du GA) et de
    générateur pour le voisinage en largeur.
    """
    rng = rng or random
    for _ in range(max_attempts):
        path: List[Connection] = []
        visited = {origin}
        current = origin
        for _ in range(max_hops):
            if current == destination:
                return path
            candidates = [c for c in net.neighbors(current, allowed_modes=allowed_modes)
                          if c.destination == destination or c.destination not in visited]
            if not candidates:
                break
            conn = rng.choice(candidates)
            path.append(conn)
            visited.add(conn.destination)
            current = conn.destination
        if current == destination:
            return path
    return None


def neighborhood_width(net: MultimodalNetwork, path: List[Connection],
                        allowed_modes: Set[Mode],
                        rng: Optional[random.Random] = None) -> Optional[List[Connection]]:
    """Voisinage en largeur (Fig. 3) : nouveau chemin aléatoire, même
    origine/destination que `path`."""
    if not path:
        return None
    origin, destination = path[0].origin, path[-1].destination
    return random_walk_path(net, origin, destination, allowed_modes,
                             max_hops=max(10, 2 * len(path)), rng=rng, max_attempts=5)


def neighborhood_depth(net: MultimodalNetwork, path: List[Connection],
                        rng: Optional[random.Random] = None) -> Optional[List[Connection]]:
    """Voisinage en profondeur (Fig. 4) : même séquence de nœuds, un arc du
    chemin est remplacé par une autre Connection disponible entre les deux
    mêmes nœuds (mode ou ligne différents), si une alternative existe.
    NB : n'a d'effet que là où plusieurs modes desservent exactement la même
    paire de nœuds consécutifs (typiquement marche/vélo/véhicule sur une
    même arête du réseau routier) ; sinon on essaie l'arc suivant du chemin.
    """
    rng = rng or random
    if not path:
        return None
    order = list(range(len(path)))
    rng.shuffle(order)
    for idx in order:
        c = path[idx]
        alternatives = [alt for alt in net.neighbors(c.origin)
                        if alt.destination == c.destination
                        and (alt.mode != c.mode or alt.line_id != c.line_id)]
        if alternatives:
            new_path = list(path)
            new_path[idx] = rng.choice(alternatives)
            return new_path
    return None


def omf(net: MultimodalNetwork, initial_path: List[Connection],
        allowed_modes: Set[Mode], nn: int = 8, it: int = 15, r: float = 0.5,
        dominance: str = "weak",
        rng: Optional[random.Random] = None,
        counter: Optional[EvalCounter] = None) -> List[Connection]:
    """Algorithme 1 (schéma fourni) : recherche locale multicritère.

    `dominance` : "weak" (Def. 2, par défaut) ou "strict" (Def. 3) — critère
    utilisé pour accepter un voisin Vi comme remplaçant de F. Paramètre
    soumis à calibration.

    `counter` : si fourni, TOUTES les évaluations de voisins passent par
    lui (pas seulement l'évaluation initiale) et la recherche s'arrête dès
    que son budget est épuisé — indispensable pour une comparaison à
    budget d'évaluations équitable avec NSGA-II/SPEA2/MOEA/D, qui n'ont pas
    de recherche locale interne consommant des évaluations "cachées".

    Retourne le chemin non dominé courant après `it` itérations de
    génération/évaluation de `nn` voisins par itération (ou moins si le
    budget de `counter` est atteint avant).
    """
    rng = rng or random
    dom_fn = strictly_dominates if dominance == "strict" else dominates
    f = initial_path
    f_obj = counter.evaluate(net, f) if counter is not None else net.evaluate_route(f).as_tuple()

    remaining = it
    while remaining > 0:
        if counter is not None and counter.exhausted():
            break
        use_width = r < rng.random()  # cf. docstring module : P(largeur) = 1 - R
        for _ in range(nn):
            if counter is not None and counter.exhausted():
                break
            vi = (neighborhood_width(net, f, allowed_modes, rng=rng) if use_width
                  else neighborhood_depth(net, f, rng=rng))
            if vi is None:
                continue
            vi_obj = counter.evaluate(net, vi) if counter is not None else net.evaluate_route(vi).as_tuple()
            if dom_fn(vi_obj, f_obj):
                f, f_obj = vi, vi_obj
        remaining -= 1
    return f
