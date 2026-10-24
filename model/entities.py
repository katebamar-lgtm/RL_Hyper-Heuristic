"""
Entités de base du réseau de transport multimodal.

Ces classes ne font AUCUN calcul d'optimisation : elles décrivent uniquement
la structure du réseau (statique). C'est volontaire : les algorithmes
d'optimisation (GA, NSGA-II, PSO, SPEA2, ...) viendront ensuite consommer
ce modèle via la classe MultimodalNetwork (voir network.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Mode(Enum):
    WALK = "walk"           # à pied
    BIKE = "bike"            # vélo
    CAR = "car"               # véhicule
    BUS = "bus"
    TRAM = "tram"
    METRO = "metro"

    @property
    def is_public_transport(self) -> bool:
        return self in (Mode.BUS, Mode.TRAM, Mode.METRO)

    @property
    def is_free_mode(self) -> bool:
        """Mode 'libre' : peut emprunter n'importe quel arc direct du réseau
        (par opposition au transport en commun, contraint aux lignes)."""
        return self in (Mode.WALK, Mode.BIKE, Mode.CAR)


@dataclass(frozen=True)
class Coordinates:
    x: float
    y: float


@dataclass
class Node:
    """Un nœud du réseau : intersection, arrêt de bus/tram/métro, POI, etc."""
    id: str
    coord: Coordinates
    name: str = ""
    is_transit_stop: bool = False  # True si au moins une ligne y passe

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Node({self.id}, {self.name})"


@dataclass
class Edge:
    """Arc direct utilisable par un mode 'libre' (marche, vélo, véhicule).

    Un Edge relie deux nœuds directement, sans passer par une ligne.
    Chaque mode libre a en général son propre sous-ensemble d'arcs
    (ex : une rue piétonne n'existe pas forcément pour la voiture).
    """
    id: str
    origin: str        # id du nœud origine
    destination: str    # id du nœud destination
    mode: Mode
    distance: float      # km
    time: float           # minutes
    cost: float            # unité monétaire (carburant, péage, amortissement vélo...)
    bidirectional: bool = True

    def __repr__(self):
        arrow = "<->" if self.bidirectional else "->"
        return f"Edge({self.origin}{arrow}{self.destination}, {self.mode.value})"


@dataclass
class LineSegment:
    """Tronçon entre deux arrêts consécutifs d'une ligne de transport en commun."""
    origin_stop: str
    destination_stop: str
    distance: float           # km
    time: float                 # minutes
    cost: float = 0.0            # coût marginal du tronçon (souvent 0, cf. TransitLine.fare)


@dataclass
class TransitLine:
    """Ligne de transport en commun (bus, tram, métro), ex : T1, T2, B1, M2.

    Une ligne est une séquence ORDONNÉE d'arrêts. On ne peut monter/descendre
    qu'aux arrêts de la ligne, et on ne peut circuler que dans l'ordre des
    arrêts (ou dans l'ordre inverse si bidirectional=True).

    Simplification demandée : pas de grille horaire. Le temps de parcours
    d'un tronçon est une valeur fixe (temps moyen), pas dépendant de l'heure.
    """
    id: str                      # ex: "T1", "B2", "M1"
    name: str
    mode: Mode                    # Mode.BUS / Mode.TRAM / Mode.METRO
    stops: List[str]               # séquence ordonnée d'ids de nœuds
    segments: List[LineSegment] = field(default_factory=list)
    fare: float = 0.0               # coût forfaitaire pour emprunter la ligne
    bidirectional: bool = True

    def __post_init__(self):
        if not self.mode.is_public_transport:
            raise ValueError(f"TransitLine {self.id}: mode {self.mode} invalide, "
                              f"attendu BUS/TRAM/METRO")
        if self.segments and len(self.segments) != len(self.stops) - 1:
            raise ValueError(f"TransitLine {self.id}: {len(self.segments)} segments "
                              f"pour {len(self.stops)} arrêts (attendu {len(self.stops)-1})")

    def stop_index(self, node_id: str) -> Optional[int]:
        try:
            return self.stops.index(node_id)
        except ValueError:
            return None

    def segment_between(self, stop_a: str, stop_b: str) -> Optional[LineSegment]:
        """Retourne le segment reliant deux arrêts consécutifs, quel que soit
        le sens de parcours (si la ligne est bidirectionnelle)."""
        ia, ib = self.stop_index(stop_a), self.stop_index(stop_b)
        if ia is None or ib is None:
            return None
        if ib == ia + 1:
            return self.segments[ia]
        if self.bidirectional and ia == ib + 1:
            seg = self.segments[ib]
            return LineSegment(stop_a, stop_b, seg.distance, seg.time, seg.cost)
        return None
