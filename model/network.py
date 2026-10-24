"""
Réseau multimodal : assemble Node / Edge / TransitLine en un graphe unique
interrogeable par les algorithmes d'optimisation.

Idée centrale : on ne travaille jamais directement avec Edge ou LineSegment
dans les algos. On construit un graphe d'adjacence homogène où chaque arête
est une `Connection` (même forme, qu'elle vienne d'un arc libre ou d'un
tronçon de ligne). C'est ce graphe que les algos (GA, NSGA-II, PSO, SPEA2...)
vont parcourir pour construire et évaluer des itinéraires.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math

from .entities import Node, Edge, TransitLine, Mode, Coordinates


@dataclass(frozen=True)
class Connection:
    """Arête générique du graphe multimodal : un 'pas' possible d'un nœud à
    un autre, quel que soit le mode."""
    origin: str
    destination: str
    mode: Mode
    time: float          # minutes
    distance: float        # km
    cost: float              # unité monétaire
    line_id: Optional[str] = None   # renseigné seulement pour BUS/TRAM/METRO

    def __repr__(self):
        via = f" via {self.line_id}" if self.line_id else ""
        return (f"{self.origin}->{self.destination} [{self.mode.value}{via}] "
                f"t={self.time:.1f}min d={self.distance:.2f}km c={self.cost:.2f}")


@dataclass
class RouteObjectives:
    """Vecteur d'objectifs d'un itinéraire complet -> ce qu'évalueront
    les algos multi-critère (4 objectifs, tous à MINIMISER)."""
    time: float = 0.0
    distance: float = 0.0
    cost: float = 0.0
    nb_transfers: int = 0   # changement de mode ou de ligne -> objectif à part entière

    def as_tuple(self) -> Tuple[float, float, float, int]:
        """Vecteur (temps, distance, coût, correspondances) — ce que
        minimisent les algos multi-objectif (GA/NSGA-II/PSO/SPEA2...)."""
        return (self.time, self.distance, self.cost, self.nb_transfers)

    def as_dict(self) -> Dict[str, float]:
        return {"time": self.time, "distance": self.distance,
                "cost": self.cost, "nb_transfers": self.nb_transfers}


class MultimodalNetwork:
    """Réseau de transport multimodal : marche, vélo, véhicule + lignes TC."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.lines: Dict[str, TransitLine] = {}
        # graphe d'adjacence construit à la demande (build_graph)
        self._adjacency: Dict[str, List[Connection]] = {}
        self._built = False

    # ------------------------------------------------------------------ #
    # Construction du réseau
    # ------------------------------------------------------------------ #

    def add_node(self, node_id: str, x: float, y: float, name: str = "",
                 is_transit_stop: bool = False) -> Node:
        node = Node(id=node_id, coord=Coordinates(x, y), name=name or node_id,
                    is_transit_stop=is_transit_stop)
        self.nodes[node_id] = node
        self._built = False
        return node

    def add_edge(self, edge_id: str, origin: str, destination: str, mode: Mode,
                 distance: Optional[float] = None, time: Optional[float] = None,
                 cost: float = 0.0, bidirectional: bool = True,
                 speed_kmh: Optional[float] = None) -> Edge:
        """Ajoute un arc pour un mode libre (WALK / BIKE / CAR).

        distance peut être omise -> calculée à vol d'oiseau depuis les coords.
        time peut être omis si `speed_kmh` est fourni -> time = distance/speed*60.
        """
        assert mode.is_free_mode, "add_edge est réservé aux modes libres (WALK/BIKE/CAR)"
        if distance is None:
            distance = self._euclidean_distance(origin, destination)
        if time is None:
            if speed_kmh is None:
                raise ValueError("Fournir soit `time`, soit `speed_kmh` pour calculer le temps")
            time = (distance / speed_kmh) * 60.0
        edge = Edge(id=edge_id, origin=origin, destination=destination, mode=mode,
                    distance=distance, time=time, cost=cost, bidirectional=bidirectional)
        self.edges.append(edge)
        self._built = False
        return edge

    def add_line(self, line: TransitLine) -> None:
        for stop in line.stops:
            if stop not in self.nodes:
                raise ValueError(f"Ligne {line.id}: arrêt inconnu '{stop}'")
            self.nodes[stop].is_transit_stop = True
        self.lines[line.id] = line
        self._built = False

    def _euclidean_distance(self, id_a: str, id_b: str) -> float:
        a, b = self.nodes[id_a].coord, self.nodes[id_b].coord
        return math.dist((a.x, a.y), (b.x, b.y))

    # ------------------------------------------------------------------ #
    # Construction du graphe d'adjacence (utilisé par les algos)
    # ------------------------------------------------------------------ #

    def build_graph(self) -> Dict[str, List[Connection]]:
        """Construit (ou reconstruit) la liste d'adjacence unifiée.

        Chaque nœud pointe vers la liste des Connection possibles depuis ce
        nœud, tous modes confondus. C'est la structure que les algos
        d'optimisation utiliseront pour explorer/décoder des itinéraires.
        """
        adjacency: Dict[str, List[Connection]] = {n: [] for n in self.nodes}

        # 1) arcs libres (marche / vélo / véhicule)
        for e in self.edges:
            adjacency[e.origin].append(Connection(e.origin, e.destination, e.mode,
                                                    e.time, e.distance, e.cost))
            if e.bidirectional:
                adjacency[e.destination].append(Connection(e.destination, e.origin, e.mode,
                                                             e.time, e.distance, e.cost))

        # 2) tronçons de lignes de transport en commun
        # NB : le coût ici est le coût MARGINAL du tronçon (souvent 0). Le
        # tarif forfaitaire (`line.fare`) n'est pas mis ici pour éviter de le
        # facturer une fois par tronçon : il est ajouté une seule fois par
        # montée dans evaluate_route().
        for line in self.lines.values():
            for i, seg in enumerate(line.segments):
                adjacency[seg.origin_stop].append(
                    Connection(seg.origin_stop, seg.destination_stop, line.mode,
                               seg.time, seg.distance, seg.cost, line_id=line.id))
                if line.bidirectional:
                    adjacency[seg.destination_stop].append(
                        Connection(seg.destination_stop, seg.origin_stop, line.mode,
                                   seg.time, seg.distance, seg.cost, line_id=line.id))

        self._adjacency = adjacency
        self._built = True
        return adjacency

    def neighbors(self, node_id: str, allowed_modes: Optional[Set[Mode]] = None
                  ) -> List[Connection]:
        """Connexions sortantes depuis un nœud, filtrables par mode.
        C'est la méthode principale utilisée par les opérateurs des algos
        (construction d'individu, mutation, décodage de chromosome...)."""
        if not self._built:
            self.build_graph()
        conns = self._adjacency.get(node_id, [])
        if allowed_modes is None:
            return conns
        return [c for c in conns if c.mode in allowed_modes]

    # ------------------------------------------------------------------ #
    # Évaluation d'un itinéraire (fonction objectif multi-critère)
    # ------------------------------------------------------------------ #

    def evaluate_route(self, route: List[Connection]) -> RouteObjectives:
        """Agrège une séquence de Connection en un vecteur d'objectifs.

        C'est LA fonction que les algos multi-critère appelleront comme
        fonction de fitness (à minimiser sur les 3 axes).
        """
        obj = RouteObjectives()
        prev_mode, prev_line = None, None
        for c in route:
            obj.time += c.time
            obj.distance += c.distance
            obj.cost += c.cost

            # Nouvelle "montée" sur une ligne TC -> facturer le tarif une fois
            boarding = c.line_id is not None and c.line_id != prev_line
            if boarding and c.line_id in self.lines:
                obj.cost += self.lines[c.line_id].fare

            changed_mode = prev_mode is not None and c.mode != prev_mode
            changed_line = c.line_id is not None and prev_line is not None and c.line_id != prev_line
            if changed_mode or changed_line:
                obj.nb_transfers += 1
            prev_mode, prev_line = c.mode, c.line_id
        return obj

    def is_valid_route(self, origin: str, destination: str,
                        route: List[Connection]) -> bool:
        """Vérifie la cohérence topologique d'un itinéraire (chaînage des
        nœuds). Utile pour valider un individu généré/muté par un algo."""
        if not route:
            return origin == destination
        if route[0].origin != origin or route[-1].destination != destination:
            return False
        for a, b in zip(route, route[1:]):
            if a.destination != b.origin:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Utilitaires
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines_by_mode = {}
        for l in self.lines.values():
            lines_by_mode.setdefault(l.mode.value, []).append(l.id)
        edges_by_mode = {}
        for e in self.edges:
            edges_by_mode.setdefault(e.mode.value, 0)
            edges_by_mode[e.mode.value] += 1
        parts = [f"Nœuds: {len(self.nodes)}"]
        parts.append("Arcs libres: " + ", ".join(f"{m}={n}" for m, n in edges_by_mode.items()))
        parts.append("Lignes TC: " + ", ".join(f"{m}={ids}" for m, ids in lines_by_mode.items()))
        return "\n".join(parts)
