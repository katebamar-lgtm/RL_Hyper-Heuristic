"""
Test de scénario SANS algorithme d'optimisation multi-critère.

Utile pour :
  - vérifier qu'un chemin existe entre deux nœuds (le réseau est bien connexe)
  - avoir une baseline mono-critère à comparer plus tard au front de Pareto
    produit par NSGA-II / SPEA2 / PSO ...

Implémentation : Dijkstra "à état". Un simple Dijkstra sur (node) ne suffit
pas dès qu'on veut un critère cohérent avec RouteObjectives, car :
  - le coût d'une Connection de transport en commun n'inclut PAS le tarif
    (celui-ci n'est facturé qu'une fois par montée, cf. network.py) ;
  - le nombre de correspondances dépend de la ligne/mode précédent.
On étend donc l'état Dijkstra à (node, mode_courant, ligne_courante), pour
rester exactement cohérent avec MultimodalNetwork.evaluate_route().
"""

from __future__ import annotations
import heapq
from typing import List, Literal, Optional, Set, Tuple

from .network import MultimodalNetwork, Connection
from .entities import Mode

Criterion = Literal["time", "distance", "cost", "transfers"]

# état = (node_id, mode_precedent_ou_None, ligne_precedente_ou_None)
State = Tuple[str, Optional[Mode], Optional[str]]


def _edge_weight(criterion: Criterion, conn: Connection,
                  prev_mode: Optional[Mode], prev_line: Optional[str],
                  net: MultimodalNetwork) -> float:
    """Coût marginal d'emprunter `conn` sachant l'état précédent.
    Reproduit EXACTEMENT la logique de MultimodalNetwork.evaluate_route()."""
    if criterion == "time":
        return conn.time
    if criterion == "distance":
        return conn.distance
    if criterion == "cost":
        w = conn.cost
        boarding = conn.line_id is not None and conn.line_id != prev_line
        if boarding and conn.line_id in net.lines:
            w += net.lines[conn.line_id].fare
        return w
    if criterion == "transfers":
        changed_mode = prev_mode is not None and conn.mode != prev_mode
        changed_line = (conn.line_id is not None and prev_line is not None
                         and conn.line_id != prev_line)
        return 1.0 if (changed_mode or changed_line) else 0.0
    raise ValueError(f"Critère inconnu: {criterion}")


def _edge_weight_combined(weights: Tuple[float, float, float, float], conn: Connection,
                           prev_mode: Optional[Mode], prev_line: Optional[str],
                           net: MultimodalNetwork) -> float:
    """Combinaison linéaire des 4 coûts marginaux (temps, distance, coût,
    correspondance), pondérée par `weights` (attendu positif, sommant à 1 —
    cf. `model/continuous_common.py::normalize_weights`). Généralisation de
    `_edge_weight` utilisée par `weighted_shortest_path` pour décoder un
    vecteur de poids continu (MOGWO/MOFA/MOABC, Phase 1) en chemin — la
    scalarisation garantit un edge weight toujours >= 0 (chaque terme est un
    produit de deux quantités positives), condition nécessaire à la
    correction de Dijkstra."""
    return (weights[0] * _edge_weight("time", conn, prev_mode, prev_line, net)
            + weights[1] * _edge_weight("distance", conn, prev_mode, prev_line, net)
            + weights[2] * _edge_weight("cost", conn, prev_mode, prev_line, net)
            + weights[3] * _edge_weight("transfers", conn, prev_mode, prev_line, net))


def _state_dijkstra(net: MultimodalNetwork, origin: str, destination: str,
                     weight_fn, allowed_modes: Optional[Set[Mode]] = None
                     ) -> Optional[List[Connection]]:
    """Dijkstra "à état" générique — factorise `shortest_path` et
    `weighted_shortest_path`, qui ne diffèrent que par la fonction de coût
    d'arête utilisée."""
    if not net._built:
        net.build_graph()
    if origin not in net.nodes or destination not in net.nodes:
        raise ValueError("Nœud origine/destination inconnu")

    start: State = (origin, None, None)
    dist = {start: 0.0}
    prev: dict = {}
    visited = set()
    heap = [(0.0, id(start), start)]
    best_goal_state = None

    while heap:
        d, _, state = heapq.heappop(heap)
        if state in visited:
            continue
        visited.add(state)

        node, prev_mode, prev_line = state
        if node == destination:
            best_goal_state = state
            break

        for conn in net.neighbors(node, allowed_modes=allowed_modes):
            w = weight_fn(conn, prev_mode, prev_line, net)
            nd = d + w
            nstate: State = (conn.destination, conn.mode, conn.line_id)
            if nstate not in dist or nd < dist[nstate]:
                dist[nstate] = nd
                prev[nstate] = (state, conn)
                heapq.heappush(heap, (nd, id(nstate), nstate))

    if best_goal_state is None:
        return None

    path: List[Connection] = []
    state = best_goal_state
    while state != start:
        pstate, conn = prev[state]
        path.append(conn)
        state = pstate
    path.reverse()
    return path


def shortest_path(net: MultimodalNetwork, origin: str, destination: str,
                   criterion: Criterion = "time",
                   allowed_modes: Optional[Set[Mode]] = None
                   ) -> Optional[List[Connection]]:
    """Dijkstra mono-critère "à état" sur le graphe multimodal.

    Retourne la liste de Connection formant le chemin optimal pour le
    critère choisi (time / distance / cost / transfers), ou None si aucun
    chemin n'existe.
    """
    return _state_dijkstra(
        net, origin, destination,
        weight_fn=lambda conn, pm, pl, n: _edge_weight(criterion, conn, pm, pl, n),
        allowed_modes=allowed_modes)


def weighted_shortest_path(net: MultimodalNetwork, origin: str, destination: str,
                            weights: Tuple[float, float, float, float],
                            allowed_modes: Optional[Set[Mode]] = None
                            ) -> Optional[List[Connection]]:
    """Dijkstra "à état" minimisant la SOMME PONDÉRÉE des 4 objectifs
    (scalarisation). Utilisé comme décodeur déterministe pour MOGWO/MOFA/
    MOABC (`model/continuous_common.py`) : chaque solution y est un vecteur
    de poids sur le simplexe, décodé en un chemin optimal pour cette
    combinaison — technique standard (apparentée à la décomposition de
    Tchebycheff de MOEA/D, cf. `model/moead.py`), qui garantit un décodage
    TOUJOURS valide (contrairement à un premier essai d'encodage par
    priorités de nœuds, abandonné.
    Limite connue et à documenter dans l'article : une scalarisation par
    somme pondérée ne peut atteindre que les points du front de Pareto
    situés sur son enveloppe convexe — les régions non convexes du front
    (le cas échéant) resteront hors de portée de MOGWO/MOFA/MOABC ainsi
    construits, quel que soit le nombre d'itérations. C'est une limite
    méthodologique à assumer explicitement, pas un bug.
    """
    return _state_dijkstra(
        net, origin, destination,
        weight_fn=lambda conn, pm, pl, n: _edge_weight_combined(weights, conn, pm, pl, n),
        allowed_modes=allowed_modes)


def describe_path(net: MultimodalNetwork, path: List[Connection]) -> str:
    if not path:
        return "(trajet vide - origine = destination)"
    obj = net.evaluate_route(path)
    lines = [f"  {c}" for c in path]
    lines.append(f"-> temps={obj.time:.1f} min | distance={obj.distance:.2f} km | "
                 f"coût={obj.cost:.2f} | correspondances={obj.nb_transfers}")
    return "\n".join(lines)
