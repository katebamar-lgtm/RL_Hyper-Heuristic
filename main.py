"""
Point d'entrée du simulateur.

Usage :
    python3 main.py                       # réseau d'exemple aléatoire (comportement historique)
    python3 main.py --source random       # idem, explicite
    python3 main.py --source data/example # importe un jeu de données réel 

Construit le réseau (généré ou importé), l'affiche graphiquement
(outputs/), et lance les 4 scénarios mono-critère (temps / distance /
coût / correspondances) pour vérifier que le modèle est cohérent --
avant de brancher les algorithmes d'optimisation multi-critère dessus.
"""

import argparse
import os

from model.scenario import shortest_path, describe_path
from model.visualization import draw_network
from examples.sample_network import build_sample_network
from model.data_loader import load_network_from_csv, load_instances_from_csv

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="random",
                         help="'random' (défaut, réseau d'exemple généré) ou chemin d'un dossier "
                              "de données (ex. data/example)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.source == "random":
        print("Source : réseau d'exemple généré aléatoirement\n")
        net = build_sample_network()
        net.build_graph()
        origin, destination = "N1", "N4"
    else:
        print(f"Source : jeu de données importé depuis '{args.source}'\n")
        net = load_network_from_csv(args.source)
        net.build_graph()
        instances = load_instances_from_csv(args.source)
        if not instances:
            raise SystemExit(f"Aucune instance trouvée dans {args.source}/instances.csv")
        origin, destination = instances[0].origin, instances[0].destination
        print(f"({len(instances)} scénario(s) disponible(s) dans instances.csv ; "
              f"utilisation du premier : {origin} -> {destination})\n")

    print(net.summary())
    print()

    overview_path = os.path.join(OUT_DIR, "network_overview.png")
    draw_network(net, title="Réseau multimodal - vue d'ensemble", save_path=overview_path)
    print(f"Image générée : {overview_path}\n")

    print(f"Scénarios de test : {origin} -> {destination} (sans algo multi-critère)\n")

    for criterion in ("time", "distance", "cost", "transfers"):
        path = shortest_path(net, origin, destination, criterion=criterion)
        print(f"--- Plus court chemin par [{criterion}] ---")
        if path is None:
            print("  Aucun chemin trouvé.")
            continue
        print(describe_path(net, path))
        fname = f"scenario_{criterion}.png"
        out_path = os.path.join(OUT_DIR, fname)
        draw_network(net, route=path, title=f"{origin} -> {destination} | critère = {criterion}",
                     save_path=out_path)
        print(f"  (image : {out_path})\n")


if __name__ == "__main__":
    main()
