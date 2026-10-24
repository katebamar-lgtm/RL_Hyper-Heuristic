"""
Charge un réseau réel depuis 4 fichiers CSV et construit un
MultimodalNetwork identique en interface à celui produit par
model/generator.py::generate_network(). Aucune modification du reste du
code n'est nécessaire : tous les algorithmes (NSGA-II, SPEA2,
MOEA/D, MOGWO, MOFA, MOABC) et le harnais de benchmark consomment ce
réseau exactement comme un réseau synthétique.
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .entities import Mode, TransitLine, LineSegment
from .network import MultimodalNetwork

MODE_MAP = {"walk": Mode.WALK, "bike": Mode.BIKE, "car": Mode.CAR,
            "bus": Mode.BUS, "tram": Mode.TRAM, "metro": Mode.METRO}


def _to_bool(v, default=True):
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "vrai", "yes", "oui")


def _to_float(v):
    return None if v in (None, "") else float(v)


def load_network_from_csv(data_dir: str) -> MultimodalNetwork:
    """Construit un MultimodalNetwork à partir de <data_dir>/{nodes,
    free_edges,lines,line_stops}.csv. Lève une ValueError explicite si un
    identifiant référencé (nœud, ligne) est introuvable."""
    net = MultimodalNetwork()

    # 1. Nœuds
    with open(os.path.join(data_dir, "nodes.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            net.add_node(row["node_id"], float(row["x"]), float(row["y"]),
                         name=row.get("name", ""))

    # 2. Arcs libres (marche/vélo/véhicule)
    edges_path = os.path.join(data_dir, "free_edges.csv")
    if os.path.exists(edges_path):
        with open(edges_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mode = MODE_MAP.get(row["mode"].strip().lower())
                if mode is None or not mode.is_free_mode:
                    raise ValueError(f"free_edges.csv: mode invalide '{row['mode']}' "
                                      f"(attendu walk/bike/car) pour l'arc {row.get('edge_id')}")
                distance = _to_float(row.get("distance_km"))
                time = _to_float(row.get("time_min"))
                speed = _to_float(row.get("speed_kmh"))
                net.add_edge(row["edge_id"], row["origin"], row["destination"], mode,
                             distance=distance, time=time,
                             cost=_to_float(row.get("cost")) or 0.0,
                             bidirectional=_to_bool(row.get("bidirectional")),
                             speed_kmh=speed)

    # 3. Lignes de transport en commun (métadonnées)
    lines_meta: Dict[str, dict] = {}
    lines_path = os.path.join(data_dir, "lines.csv")
    if os.path.exists(lines_path):
        with open(lines_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lines_meta[row["line_id"]] = row

    # 4. Séquences d'arrêts + segments, groupées par ligne
    stops_path = os.path.join(data_dir, "line_stops.csv")
    if os.path.exists(stops_path):
        by_line: Dict[str, List[dict]] = {}
        with open(stops_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_line.setdefault(row["line_id"], []).append(row)

        for line_id, rows in by_line.items():
            if line_id not in lines_meta:
                raise ValueError(f"line_stops.csv référence la ligne inconnue '{line_id}' "
                                  f"(absente de lines.csv)")
            rows.sort(key=lambda r: int(r["sequence"]))
            stops = [r["stop_id"] for r in rows]
            segments = []
            for i in range(len(rows) - 1):
                d = _to_float(rows[i].get("seg_distance_km"))
                t = _to_float(rows[i].get("seg_time_min"))
                if d is None or t is None:
                    raise ValueError(f"line_stops.csv: segment manquant pour la ligne "
                                      f"'{line_id}' entre '{stops[i]}' et '{stops[i+1]}' "
                                      f"(seg_distance_km/seg_time_min requis sauf dernier arrêt)")
                segments.append(LineSegment(stops[i], stops[i + 1], d, t,
                                             _to_float(rows[i].get("seg_cost")) or 0.0))

            meta = lines_meta[line_id]
            mode = MODE_MAP.get(meta["mode"].strip().lower())
            if mode is None or not mode.is_public_transport:
                raise ValueError(f"lines.csv: mode invalide '{meta['mode']}' "
                                  f"(attendu bus/tram/metro) pour la ligne {line_id}")
            net.add_line(TransitLine(
                id=line_id, name=meta.get("name", "") or line_id, mode=mode,
                stops=stops, segments=segments,
                fare=_to_float(meta.get("fare")) or 0.0,
                bidirectional=_to_bool(meta.get("bidirectional")),
            ))

    return net


@dataclass
class RealInstance:
    name: str
    origin: str
    destination: str
    population_size: int
    budget: int


def load_instances_from_csv(data_dir: str) -> List[RealInstance]:
   
    path = os.path.join(data_dir, "instances.csv")
    instances = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            instances.append(RealInstance(
                name=row["name"], origin=row["origin"], destination=row["destination"],
                population_size=int(row["population_size"]), budget=int(row["budget"]),
            ))
    return instances
