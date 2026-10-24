"""
Fusionne les arrêts GTFS (nodes_gtfs_stops.csv, produit par
scripts/import_gtfs.py) dans le réseau piéton/vélo/véhicule (nodes.csv/
free_edges.csv, produit par scripts/import_osm.py), et ajoute un arc de
marche entre chaque arrêt GTFS et le nœud OSM piéton le plus proche --
sans quoi les lignes de transport en commun resteraient des îlots
déconnectés du reste du réseau (cf. avertissement affiché par
import_gtfs.py).

Usage :
    python3 scripts/merge_gtfs_osm.py data/lyon
    python3 scripts/merge_gtfs_osm.py data/lyon --max-access-km 0.3

Suppose que <data_dir> contient déjà nodes.csv + free_edges.csv (issus
de import_osm.py) ET nodes_gtfs_stops.csv (+ line_stops.csv, issus de
import_gtfs.py). Écrase nodes.csv/free_edges.csv/line_stops.csv EN
PLACE -- une sauvegarde .bak de chacun est créée avant toute écriture.

Pour éviter toute collision d'identifiant entre un arrêt GTFS et un
nœud OSM (les deux utilisent souvent des identifiants numériques
courts), tous les identifiants d'arrêts GTFS sont systématiquement
préfixés par "GTFS_" -- y compris dans line_stops.csv, mis à jour en
conséquence.
"""
import argparse
import csv
import math
import os
import shutil
import sys


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _backup(path):
    if os.path.exists(path):
        shutil.copy(path, path + ".bak")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--max-access-km", type=float, default=1.0,
                         help="distance (km) au-delà de laquelle un avertissement est affiché "
                              "pour un arrêt trop loin du réseau piéton (défaut 1.0)")
    parser.add_argument("--speed-kmh", type=float, default=4.5, help="vitesse de marche pour l'arc d'accès (défaut 4.5)")
    args = parser.parse_args()

    nodes_path = os.path.join(args.data_dir, "nodes.csv")
    edges_path = os.path.join(args.data_dir, "free_edges.csv")
    gtfs_nodes_path = os.path.join(args.data_dir, "nodes_gtfs_stops.csv")
    line_stops_path = os.path.join(args.data_dir, "line_stops.csv")

    for p in (nodes_path, edges_path, gtfs_nodes_path):
        if not os.path.exists(p):
            sys.exit(f"Fichier manquant : {p}\n"
                      f"(lancez d'abord import_osm.py ET import_gtfs.py sur ce même dossier)")

    with open(nodes_path, newline="", encoding="utf-8") as f:
        osm_nodes = list(csv.DictReader(f))
    with open(gtfs_nodes_path, newline="", encoding="utf-8") as f:
        gtfs_nodes = list(csv.DictReader(f))

    # prefixage systematique -> aucune ambiguite possible avec les IDs OSM
    id_map = {}
    for r in gtfs_nodes:
        new_id = "GTFS_" + r["node_id"]
        id_map[r["node_id"]] = new_id
        r["node_id"] = new_id

    print(f"{len(osm_nodes)} nœuds OSM + {len(gtfs_nodes)} arrêts GTFS (préfixés 'GTFS_') à fusionner.")

    # --- fusion des noeuds ---
    _backup(nodes_path)
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["node_id", "x", "y", "name"])
        w.writeheader()
        for r in osm_nodes:
            w.writerow({"node_id": r["node_id"], "x": r["x"], "y": r["y"], "name": r.get("name", "")})
        for r in gtfs_nodes:
            w.writerow({"node_id": r["node_id"], "x": r["x"], "y": r["y"], "name": r.get("name", "")})

    new_edges = []
    warnings = []
    for g in gtfs_nodes:
        gx, gy = float(g["x"]), float(g["y"])
        best_id, best_d = None, None
        for o in osm_nodes:
            ox, oy = float(o["x"]), float(o["y"])
            d = haversine_km(gy, gx, oy, ox)
            if best_d is None or d < best_d:
                best_id, best_d = o["node_id"], d
        if best_id is None:
            continue
        if best_d > args.max_access_km:
            warnings.append((g["node_id"], best_id, round(best_d, 3)))
        new_edges.append({"origin": g["node_id"], "destination": best_id, "mode": "walk",
                           "distance_km": round(best_d, 4), "speed_kmh": args.speed_kmh,
                           "cost": 0.0, "bidirectional": "true"})

    # --- ajout des arcs d'acces a free_edges.csv ---
    _backup(edges_path)
    with open(edges_path, newline="", encoding="utf-8") as f:
        existing_edges = list(csv.DictReader(f))
        edge_fields = list(existing_edges[0].keys()) if existing_edges else \
            ["edge_id", "origin", "destination", "mode", "distance_km", "speed_kmh", "cost", "bidirectional"]
    if "edge_id" not in edge_fields:
        edge_fields = ["edge_id"] + edge_fields
    max_eid = 0
    for r in existing_edges:
        try:
            max_eid = max(max_eid, int(''.join(filter(str.isdigit, r.get("edge_id", "0"))) or 0))
        except ValueError:
            pass
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=edge_fields)
        w.writeheader()
        for r in existing_edges:
            w.writerow({k: r.get(k, "") for k in edge_fields})
        for i, r in enumerate(new_edges, start=1):
            row = {k: r.get(k, "") for k in edge_fields}
            row["edge_id"] = f"EACC{max_eid + i}"
            w.writerow(row)

    # --- mise a jour des references dans line_stops.csv (IDs prefixes) ---
    if os.path.exists(line_stops_path):
        _backup(line_stops_path)
        with open(line_stops_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["stop_id"] in id_map:
                r["stop_id"] = id_map[r["stop_id"]]
        with open(line_stops_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"line_stops.csv mis à jour ({len(rows)} lignes, identifiants préfixés).")

    print(f"\n{len(new_edges)} arcs d'accès (marche) ajoutés, reliant chaque arrêt GTFS au nœud OSM le plus proche.")
    if warnings:
        print(f"\nATTENTION : {len(warnings)} arrêt(s) à plus de {args.max_access_km} km du nœud piéton le "
              f"plus proche (zone OSM téléchargée probablement trop petite pour couvrir ces arrêts) :")
        for stop_id, node_id, dist in warnings[:15]:
            print(f"   {stop_id} -> {node_id} ({dist} km)")
        print("Envisagez de relancer import_osm.py avec un --radius plus grand si ces arrêts sont importants.")
    print(f"\nFusion terminée. Fichiers modifiés : {nodes_path}, {edges_path}"
          + (f", {line_stops_path}" if os.path.exists(line_stops_path) else ""))
    print("Sauvegardes .bak créées avant modification.")


if __name__ == "__main__":
    main()
