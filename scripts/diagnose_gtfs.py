"""
Diagnostique un fichier GTFS pour comprendre pourquoi certains modes
(souvent tram/métro) n'apparaissent pas dans l'extraction de
scripts/import_gtfs.py -- inspecte routes.txt/trips.txt/stop_times.txt
sans rien écrire, juste des comptages à chaque étape du filtrage.

Usage :
    python3 scripts/diagnose_gtfs.py lyon_gtfs.zip
"""
import argparse
import csv
import io
import sys
import zipfile
from collections import Counter, defaultdict


def _read(zf, name):
    with zf.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gtfs_zip")
    args = parser.parse_args()

    with zipfile.ZipFile(args.gtfs_zip) as zf:
        names = zf.namelist()
        print(f"Fichiers présents dans l'archive : {names}\n")
        required = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
        missing = [r for r in required if r not in names]
        if missing:
            sys.exit(f"ERREUR : fichier(s) GTFS manquant(s) dans l'archive : {missing}")

        routes = _read(zf, "routes.txt")
        trips = _read(zf, "trips.txt")
        stop_times = _read(zf, "stop_times.txt")

    print("=== 1. Distribution des route_type dans routes.txt ===")
    type_counts = Counter(r.get("route_type", "MANQUANT") for r in routes)
    for t, c in sorted(type_counts.items(), key=lambda x: str(x[0])):
        label = {"0": "tram", "1": "métro", "2": "train", "3": "bus", "4": "ferry",
                  "5": "câble", "6": "téléphérique", "7": "funiculaire"}.get(str(t), "??? (non standard)")
        print(f"  route_type={t!r:>10s}  ({label:15s}) : {c} lignes")
    print("\n  -> Si 0 et 1 n'apparaissent PAS ici, le tram/métro de ce réseau utilise "
          "un code route_type différent dans CE fichier (variantes non-standard existent) "
          "-- regardez route_short_name/route_long_name ci-dessous pour identifier le bon code.")

    print("\n=== 2. Exemples de lignes par route_type (pour identifier tram/métro si code non standard) ===")
    by_type = defaultdict(list)
    for r in routes:
        by_type[r.get("route_type", "?")].append(r.get("route_short_name") or r.get("route_long_name", "?"))
    for t, names_list in sorted(by_type.items(), key=lambda x: str(x[0])):
        sample = names_list[:8]
        print(f"  type {t!r} ({len(names_list)} lignes) : {sample}{' ...' if len(names_list) > 8 else ''}")

    print("\n=== 3. Combien de routes de chaque type ont au moins UN trip dans trips.txt ? ===")
    route_ids_with_trips = {t["route_id"] for t in trips}
    for t in sorted(type_counts):
        route_ids_of_type = {r["route_id"] for r in routes if r.get("route_type") == t}
        n_with_trips = len(route_ids_of_type & route_ids_with_trips)
        print(f"  route_type={t!r:>10s} : {n_with_trips}/{len(route_ids_of_type)} lignes ont >=1 trip")

    print("\n=== 4. Combien de trips ont >=2 lignes dans stop_times.txt (condition requise par import_gtfs.py) ? ===")
    st_counts = Counter(st["trip_id"] for st in stop_times)
    trip_to_route = {t["trip_id"]: t["route_id"] for t in trips}
    route_to_valid_trips = defaultdict(int)
    for trip_id, n_stops in st_counts.items():
        if n_stops >= 2 and trip_id in trip_to_route:
            route_to_valid_trips[trip_to_route[trip_id]] += 1
    for t in sorted(type_counts):
        route_ids_of_type = {r["route_id"] for r in routes if r.get("route_type") == t}
        n_valid = sum(1 for rid in route_ids_of_type if route_to_valid_trips.get(rid, 0) > 0)
        print(f"  route_type={t!r:>10s} : {n_valid}/{len(route_ids_of_type)} lignes ont un trip exploitable "
              f"(>=2 stop_times) -- ce sont EXACTEMENT celles qu'écrira import_gtfs.py")

    print("\n=== Conclusion ===")
    for target, label in [("0", "tram"), ("1", "métro")]:
        route_ids_of_type = {r["route_id"] for r in routes if r.get("route_type") == target}
        n_valid = sum(1 for rid in route_ids_of_type if route_to_valid_trips.get(rid, 0) > 0)
        if not route_ids_of_type:
            print(f"  {label} (route_type={target}) : AUCUNE ligne avec ce code dans routes.txt -- "
                  f"ce réseau encode probablement {label} avec un autre route_type (voir §2 ci-dessus).")
        elif n_valid == 0:
            print(f"  {label} (route_type={target}) : {len(route_ids_of_type)} ligne(s) déclarée(s) mais "
                  f"AUCUNE n'a de trip exploitable -- fichier GTFS partiel/horaires manquants pour ce mode.")
        else:
            print(f"  {label} (route_type={target}) : {n_valid} ligne(s) exploitable(s) -- "
                  f"devrait apparaître dans l'extraction, vérifiez --max-lines.")


if __name__ == "__main__":
    main()
