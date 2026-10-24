"""
Exemple de construction d'un petit réseau multimodal, pour valider le modèle
avant de brancher les algos d'optimisation dessus.

Scénario jouet :
  - 8 nœuds (N1..N8), dont certains sont des arrêts de transport en commun.
  - Marche et vélo : réseau assez dense (beaucoup d'arcs directs).
  - Véhicule : réseau routier plus restreint.
  - 2 lignes de bus (B1, B2), 1 ligne de tram (T1).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.entities import Mode, TransitLine, LineSegment
from model.network import MultimodalNetwork


def build_sample_network() -> MultimodalNetwork:
    net = MultimodalNetwork()

    # --- Nœuds (coordonnées arbitraires en km) ---
    coords = {
        "N1": (0, 0), "N2": (1, 0.5), "N3": (2, 0), "N4": (3, 1),
        "N5": (1, 2), "N6": (2.5, 2.2), "N7": (0.5, 3), "N8": (3.5, 3),
    }
    for nid, (x, y) in coords.items():
        net.add_node(nid, x, y)

    # --- Arcs "marche" (vitesse ~5 km/h), réseau dense ---
    walk_pairs = [("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N1", "N5"),
                  ("N5", "N6"), ("N6", "N4"), ("N5", "N7"), ("N6", "N8"),
                  ("N7", "N8"), ("N2", "N5"), ("N3", "N6")]
    for i, (a, b) in enumerate(walk_pairs):
        net.add_edge(f"W{i}", a, b, Mode.WALK, speed_kmh=5.0, cost=0.0)

    # --- Arcs "vélo" (vitesse ~15 km/h), mêmes liaisons + coût symbolique ---
    for i, (a, b) in enumerate(walk_pairs):
        net.add_edge(f"K{i}", a, b, Mode.BIKE, speed_kmh=15.0, cost=0.2)

    # --- Arcs "véhicule" (vitesse ~40 km/h), réseau routier plus restreint ---
    car_pairs = [("N1", "N3"), ("N3", "N4"), ("N1", "N7"), ("N7", "N8"),
                 ("N4", "N8")]
    for i, (a, b) in enumerate(car_pairs):
        net.add_edge(f"C{i}", a, b, Mode.CAR, speed_kmh=40.0, cost=1.5)

    # --- Ligne de bus B1 : N1 -> N2 -> N3 -> N4 ---
    b1_stops = ["N1", "N2", "N3", "N4"]
    b1_segments = [
        LineSegment("N1", "N2", distance=1.2, time=4.0),
        LineSegment("N2", "N3", distance=1.1, time=3.5),
        LineSegment("N3", "N4", distance=1.4, time=4.5),
    ]
    net.add_line(TransitLine("B1", "Bus Ligne 1", Mode.BUS, b1_stops, b1_segments, fare=1.0))

    # --- Ligne de bus B2 : N7 -> N5 -> N6 -> N8 ---
    b2_stops = ["N7", "N5", "N6", "N8"]
    b2_segments = [
        LineSegment("N7", "N5", distance=1.3, time=4.0),
        LineSegment("N5", "N6", distance=1.6, time=5.0),
        LineSegment("N6", "N8", distance=1.2, time=3.8),
    ]
    net.add_line(TransitLine("B2", "Bus Ligne 2", Mode.BUS, b2_stops, b2_segments, fare=1.0))

    # --- Ligne de tram T1 : N1 -> N5 -> N6 -> N4 (plus rapide, plus cher) ---
    t1_stops = ["N1", "N5", "N6", "N4"]
    t1_segments = [
        LineSegment("N1", "N5", distance=2.2, time=5.0),
        LineSegment("N5", "N6", distance=1.6, time=3.5),
        LineSegment("N6", "N4", distance=1.8, time=4.0),
    ]
    net.add_line(TransitLine("T1", "Tram Ligne 1", Mode.TRAM, t1_stops, t1_segments, fare=1.5))

    return net


if __name__ == "__main__":
    net = build_sample_network()
    net.build_graph()

    print(net.summary())
    print()

    print("Connexions sortantes depuis N1 :")
    for c in net.neighbors("N1"):
        print(" ", c)
    print()

    # Exemple d'itinéraire manuel : N1 --(tram T1)--> N5 --(bus B2)--> N6 --(marche)--> N4... non
    # on construit un itinéraire simple et cohérent : N1 -[tram T1]-> N6 -[marche]-> N4
    route = [c for c in net.neighbors("N1") if c.destination == "N5" and c.line_id == "T1"] \
          + [c for c in net.neighbors("N5") if c.destination == "N6" and c.line_id == "T1"] \
          + [c for c in net.neighbors("N6") if c.destination == "N4" and c.mode == Mode.WALK]

    print("Itinéraire test : N1 -> N5 -> N6 (tram T1) puis N6 -> N4 (marche)")
    for c in route:
        print(" ", c)

    valid = net.is_valid_route("N1", "N4", route)
    obj = net.evaluate_route(route)
    print(f"\nItinéraire valide: {valid}")
    print(f"Objectifs -> temps={obj.time:.1f} min | distance={obj.distance:.2f} km | "
          f"coût={obj.cost:.2f} | correspondances={obj.nb_transfers}")
