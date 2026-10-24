"""
Extrait les lignes de transport en commun d'un flux GTFS (fichier .zip
standard) et les écrit au format attendu par model/data_loader.py
(lines.csv, line_stops.csv), en complément des nodes.csv/free_edges.csv
produits par scripts/import_osm.py.

N'utilise que la bibliothèque standard (zipfile + csv), aucune dépendance
GTFS externe requise.

Usage :
    python3 scripts/import_gtfs.py lyon_gtfs.zip data/lyon --modes 0,1,3

`--modes` : route_type(s) GTFS à garder (cf. spec GTFS routes.txt) :
    0 = tram, 1 = métro, 3 = bus, 2 = train, 4 = ferry, 5 = câble, 6 = téléphérique, 7 = funiculaire
Par défaut : 0,1,3 (tram, métro, bus -- les trois modes du projet).

IMPORTANT -- absence de grille horaire dans le modèle (cf. TransitLine) :
ce script moyenne le temps de trajet entre arrêts consécutifs sur TOUS
les passages (`trips`) d'une même ligne trouvés dans stop_times.txt,
plutôt que de garder un horaire précis. C'est une approximation
délibérée, cohérente avec la simplification déjà assumée dans le modèle.

Les arrêts GTFS (stops.txt) sont ajoutés comme des NŒUDS SUPPLÉMENTAIRES
au réseau -- il faut ensuite les relier au réseau piéton (OSM) par des
arcs de marche "d'accès" (cf. fonction `suggest_access_edges` ci-dessous),
sans quoi les lignes de transport en commun resteraient des îlots
déconnectés du reste du réseau.
"""
import argparse
import csv
import io
import math
import os
import zipfile
from collections import defaultdict

DEFAULT_MODES = {0, 1, 3}
GTFS_MODE_TO_PROJECT = {0: "tram", 1: "metro", 3: "bus"}


def _read_gtfs_csv(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig")
        return list(csv.DictReader(text))


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _gtfs_time_to_seconds(t: str):
    """GTFS autorise des heures > 24 (ex. 25:10:00 pour un service après minuit)."""
    h, m, s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gtfs_zip", help="Chemin du fichier GTFS .zip")
    parser.add_argument("out_dir", help="Dossier de sortie, ex. data/lyon (doit déjà contenir nodes.csv si fusion avec OSM)")
    parser.add_argument("--modes", default="0,1,3", help="route_type GTFS à garder, séparés par virgule (défaut 0,1,3)")
    parser.add_argument("--max-lines", type=int, default=15,
                         help="Limite le nombre de lignes exportées (défaut 15, garder le réseau petit)")
    args = parser.parse_args()
    keep_route_types = {int(x) for x in args.modes.split(",")}

    os.makedirs(args.out_dir, exist_ok=True)

    with zipfile.ZipFile(args.gtfs_zip) as zf:
        stops = {r["stop_id"]: r for r in _read_gtfs_csv(zf, "stops.txt")}
        routes = {r["route_id"]: r for r in _read_gtfs_csv(zf, "routes.txt")
                  if int(r.get("route_type", -1)) in keep_route_types}
        trips = [r for r in _read_gtfs_csv(zf, "trips.txt") if r["route_id"] in routes]
        trip_to_route = {r["trip_id"]: r["route_id"] for r in trips}
        stop_times = _read_gtfs_csv(zf, "stop_times.txt")

    print(f"{len(routes)} lignes retenues (route_type in {keep_route_types}), "
          f"{len(stops)} arrêts GTFS, {len(trips)} trips, {len(stop_times)} stop_times.")

    if len(routes) > args.max_lines:
        # Garder EN PRIORITÉ les lignes tram/métro (route_type != 3) : rares
        # mais structurantes pour un réseau multimodal. Compléter avec des
        # lignes de bus jusqu'à la limite. Sans cette priorité, un simple
        # "les N premières lignes du fichier" peut ne retenir AUCUNE ligne
        # tram/métro si routes.txt liste les bus en premier (cas rencontré
        # sur un export réel -- corrigé après un premier essai à 0 tram/métro).
        non_bus_ids = [k for k, v in routes.items() if int(v.get("route_type", 3)) != 3]
        bus_ids = [k for k, v in routes.items() if int(v.get("route_type", 3)) == 3]
        remaining = max(0, args.max_lines - len(non_bus_ids))
        kept_ids = set(non_bus_ids) | set(bus_ids[:remaining])
        if len(non_bus_ids) > args.max_lines:
            print(f"ATTENTION : {len(non_bus_ids)} lignes tram/métro à elles seules dépassent "
                  f"--max-lines={args.max_lines} -- toutes conservées quand même (aucune ligne de bus "
                  f"retenue). Augmentez --max-lines si vous voulez aussi des bus.")
        print(f"Limitation à {len(kept_ids)} lignes sur {len(routes)} "
              f"({len(non_bus_ids) if len(non_bus_ids) <= args.max_lines else args.max_lines} tram/métro "
              f"conservées en priorité, reste en bus) -- ajustez --max-lines si besoin.")
        routes = {k: v for k, v in routes.items() if k in kept_ids}
        trips = [t for t in trips if t["route_id"] in routes]
        trip_to_route = {r["trip_id"]: r["route_id"] for r in trips}

    # regrouper stop_times par trip, trie par stop_sequence
    by_trip = defaultdict(list)
    kept_trip_ids = set(trip_to_route)
    for st in stop_times:
        if st["trip_id"] in kept_trip_ids:
            by_trip[st["trip_id"]].append(st)
    for trip_id in by_trip:
        by_trip[trip_id].sort(key=lambda r: int(r["stop_sequence"]))

    # pour chaque ligne : choisir la sequence d'arrets du trip le plus long
    # (heuristique simple -- un GTFS a souvent 2 variantes/sens par ligne),
    # et moyenner le temps entre arrets consecutifs sur TOUS les trips de
    # cette ligne qui suivent la MEME sequence d'arrets.
    route_to_trips = defaultdict(list)
    for trip_id, route_id in trip_to_route.items():
        if trip_id in by_trip and len(by_trip[trip_id]) >= 2:
            route_to_trips[route_id].append(trip_id)

    lines_rows = []
    line_stops_rows = []
    used_stop_ids = set()

    for route_id, trip_ids in route_to_trips.items():
        # trip de reference = le plus long (le plus de stops)
        ref_trip = max(trip_ids, key=lambda tid: len(by_trip[tid]))
        ref_sequence = [st["stop_id"] for st in by_trip[ref_trip]]

        # temps moyen entre chaque paire d'arrets consecutifs, sur tous les
        # trips qui suivent EXACTEMENT cette sequence
        n = len(ref_sequence)
        seg_times = [[] for _ in range(n - 1)]
        for tid in trip_ids:
            seq = [st["stop_id"] for st in by_trip[tid]]
            if seq != ref_sequence:
                continue
            times = [_gtfs_time_to_seconds(st["arrival_time"]) for st in by_trip[tid]]
            for i in range(n - 1):
                seg_times[i].append(times[i + 1] - times[i])

        route = routes[route_id]
        mode = GTFS_MODE_TO_PROJECT.get(int(route["route_type"]), "bus")
        line_id = route.get("route_short_name") or route_id
        lines_rows.append({"line_id": line_id, "name": route.get("route_long_name", ""),
                            "mode": mode, "fare": "", "bidirectional": "true"})

        for i, stop_id in enumerate(ref_sequence):
            used_stop_ids.add(stop_id)
            row = {"line_id": line_id, "sequence": i, "stop_id": stop_id,
                   "seg_distance_km": "", "seg_time_min": "", "seg_cost": 0}
            if i < n - 1:
                s_a, s_b = stops[ref_sequence[i]], stops[ref_sequence[i + 1]]
                dist_km = haversine_km(float(s_a["stop_lat"]), float(s_a["stop_lon"]),
                                        float(s_b["stop_lat"]), float(s_b["stop_lon"]))
                valid_times = [t for t in seg_times[i] if t > 0]
                avg_min = (sum(valid_times) / len(valid_times) / 60.0) if valid_times else (dist_km / 20.0 * 60)
                row["seg_distance_km"] = round(dist_km, 4)
                row["seg_time_min"] = round(avg_min, 2)
            line_stops_rows.append(row)

    lines_path = os.path.join(args.out_dir, "lines.csv")
    with open(lines_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line_id", "name", "mode", "fare", "bidirectional"])
        w.writeheader()
        w.writerows(lines_rows)
    print(f"{len(lines_rows)} lignes -> {lines_path} (fare vide -- à compléter manuellement)")

    stops_path = os.path.join(args.out_dir, "line_stops.csv")
    with open(stops_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line_id", "sequence", "stop_id",
                                           "seg_distance_km", "seg_time_min", "seg_cost"])
        w.writeheader()
        w.writerows(line_stops_rows)
    print(f"{len(line_stops_rows)} arrêts (lignes x séquence) -> {stops_path}")

    # arrets GTFS a ajouter comme noeuds -- NE remplace PAS un nodes.csv
    # existant (ex. issu d'OSM) : ecrit un fichier separe a fusionner
    # manuellement, pour eviter d'ecraser des donnees deja presentes.
    gtfs_nodes_path = os.path.join(args.out_dir, "nodes_gtfs_stops.csv")
    with open(gtfs_nodes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x", "y", "name"])
        for stop_id in used_stop_ids:
            s = stops[stop_id]
            w.writerow([stop_id, s["stop_lon"], s["stop_lat"], s.get("stop_name", "")])
    print(f"{len(used_stop_ids)} arrêts -> {gtfs_nodes_path}")
    print("\nATTENTION : ce fichier est SÉPARÉ de nodes.csv -- fusionnez-le manuellement "
          "(ou avec un script) si vous avez déjà un nodes.csv issu d'OSM, en évitant les "
          "doublons d'identifiants. Sans arcs de marche reliant ces arrêts au reste du "
          "réseau (nodes.csv/free_edges.csv), les lignes de transport en commun resteront "
          "des îlots déconnectés -- ajoutez des arcs 'walk' entre chaque arrêt GTFS et le "
          "nœud OSM piéton le plus proche avant de charger le réseau complet.")


if __name__ == "__main__":
    main()
