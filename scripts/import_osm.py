"""
Extrait un réseau piéton/vélo/routier depuis OpenStreetMap (via osmnx) et
l'écrit au format attendu par model/data_loader.py (nodes.csv,
free_edges.csv).

Nécessite une connexion internet (interroge l'API Overpass d'OSM) et :
    pip install osmnx --break-system-packages

Usage :
    python3 scripts/import_osm.py "Oran, Algeria" data/oran --radius 1500
    python3 scripts/import_osm.py "Lyon, France" data/lyon --radius 2000

`--radius` (mètres) limite la zone autour du centre du lieu donné --
IMPORTANT : rester sur un quartier plutôt qu'une ville entière, le
nombre de nœuds doit rester compatible avec les algorithmes (cf.
MAX_NODES=80 dans webapp/server.py ; au-delà, le calcul devient lent.

Vitesses par défaut (km/h) utilisées pour calculer le temps à partir de
la distance -- à ajuster selon le contexte local :
"""
import argparse
import csv
import os
import sys

SPEED_KMH = {"walk": 4.5, "bike": 15.0, "car": 30.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("place", help='Nom du lieu, ex. "Oran, Algeria"')
    parser.add_argument("out_dir", help="Dossier de sortie, ex. data/oran")
    parser.add_argument("--radius", type=int, default=1500, help="Rayon en mètres autour du centre (défaut 1500)")
    parser.add_argument("--max-nodes", type=int, default=80,
                         help="Sous-échantillonne si le graphe dépasse ce nombre de nœuds (défaut 80)")
    args = parser.parse_args()

    try:
        import osmnx as ox
    except ImportError:
        sys.exit("osmnx non installé : pip install osmnx --break-system-packages")

    os.makedirs(args.out_dir, exist_ok=True)

    center = ox.geocode(args.place)
    print(f"Centre géocodé pour '{args.place}' : {center}")

    all_nodes = {}   # osmid -> (x, y)
    all_edges = []   # (origin, destination, mode, distance_km, bidirectional)

    for mode, network_type in [("walk", "walk"), ("bike", "bike"), ("car", "drive")]:
        print(f"Téléchargement du réseau '{network_type}' (rayon {args.radius} m)...")
        G = ox.graph_from_point(center, dist=args.radius, network_type=network_type, simplify=True)

        for node_id, data in G.nodes(data=True):
            all_nodes[node_id] = (data["x"], data["y"])

        if mode == "car":
            # Réseau routier : garder la direction telle que fournie par OSM
            # (respecte les sens uniques) -- ne PAS forcer bidirectional=True.
            for u, v, data in G.edges(data=True):
                length_km = data.get("length", 0) / 1000.0
                all_edges.append((u, v, mode, length_km, False))
        else:
            # Marche/vélo : réseau non orienté -- dédupliquer pour n'écrire
            # qu'UNE ligne par paire de nœuds (bidirectional=True génère déjà
            # les deux sens dans model/network.py::build_graph ; écrire aussi
            # les deux directions ici les doublerait).
            G = ox.convert.to_undirected(G)
            seen_pairs = set()
            for u, v, data in G.edges(data=True):
                key = tuple(sorted((u, v)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                length_km = data.get("length", 0) / 1000.0
                all_edges.append((u, v, mode, length_km, True))

    # sous-echantillonnage si trop de noeuds (garde les plus connectes)
    if len(all_nodes) > args.max_nodes:
        print(f"{len(all_nodes)} nœuds > --max-nodes={args.max_nodes} : réduction du rayon recommandée. "
              f"Écriture limitée aux {args.max_nodes} premiers nœuds rencontrés (résultat partiel, "
              f"le réseau peut être déconnecté -- préférez relancer avec --radius plus petit).")
        keep = set(list(all_nodes.keys())[:args.max_nodes])
        all_nodes = {k: v for k, v in all_nodes.items() if k in keep}
        all_edges = [e for e in all_edges if e[0] in keep and e[1] in keep]

    node_id_map = {osmid: f"N{i+1}" for i, osmid in enumerate(all_nodes)}

    nodes_path = os.path.join(args.out_dir, "nodes.csv")
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x", "y", "name"])
        for osmid, (x, y) in all_nodes.items():
            w.writerow([node_id_map[osmid], x, y, ""])
    print(f"{len(all_nodes)} nœuds -> {nodes_path}")

    edges_path = os.path.join(args.out_dir, "free_edges.csv")
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "origin", "destination", "mode", "distance_km", "speed_kmh", "cost", "bidirectional"])
        eid = 0
        for u, v, mode, dist_km, bidir in all_edges:
            if u not in node_id_map or v not in node_id_map:
                continue
            eid += 1
            cost = 0.0 if mode in ("walk", "bike") else round(dist_km * 0.15, 3)  # ex. carburant, à ajuster
            w.writerow([f"E{eid}", node_id_map[u], node_id_map[v], mode, round(dist_km, 4),
                        SPEED_KMH[mode], cost, "true" if bidir else "false"])
    print(f"{eid} arcs -> {edges_path}")
    print(f"\nVitesses utilisées pour calculer time_min à partir de distance_km : {SPEED_KMH} "
          f"(colonne speed_kmh, cf. model/data_loader.py -- ajustez ces valeurs dans le fichier "
          f"si besoin, une valeur par ligne).")
    print("\nRappel : lines.csv/line_stops.csv (transport en commun) ne sont PAS produits par ce "
          "script -- utilisez scripts/import_gtfs.py si un flux GTFS existe pour ce réseau, "
          "ou saisissez-les manuellement.")


if __name__ == "__main__":
    main()
