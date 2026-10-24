"""
Verification empirique de la regle leave-family-out pour OMF-statique,
OMF-aleatoire et OMF-RL -- ces trois variantes partagent les memes
operateurs de voisinage (neighborhood_width/neighborhood_depth,
model/omf.py) ENTRE ELLES : verifie si un regroupement de famille est
necessaire pour le calcul du front de reference (IGD/GD).

Usage : python verify_leave_family_out_omf.py
(a lancer a la racine du depot, results/phase4_raw_runs.jsonl doit exister
et contenir des runs pour OMF-statique/OMF-aleatoire/OMF-RL)
"""
import json
from collections import defaultdict

RAW_RUNS_PATH = "results/phase4_raw_runs.jsonl"

ALL_ALGOS = ['NSGA-II', 'SPEA2', 'MOEA/D', 'MOGWO', 'MOFA', 'MOABC',
             'RL-MOEA/D', 'RL-MOEA/D-random', 'OMF-statique', 'OMF-aleatoire', 'OMF-RL']
CANDIDATE_OMF_FAMILY = ['OMF-statique', 'OMF-aleatoire', 'OMF-RL']


def load_points(path):
    points = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            points[d["instance"]][d["algo"]].extend(tuple(o) for o in d["objectives"])
    return points


def overlap_fraction(pts_a, union_b):
    set_a = set(pts_a)
    if not set_a:
        return None
    return sum(1 for p in set_a if p in union_b) / len(set_a)


def family_overlap(points, algo, family, instances):
    within_vals, cross_vals = [], []
    for inst in instances:
        algos_here = points[inst]
        if algo not in algos_here:
            continue
        pts_a = algos_here[algo]
        others_family = [a for a in family if a != algo and a in algos_here]
        others_cross = [a for a in ALL_ALGOS if a not in family and a in algos_here]
        if others_family:
            union_f = set()
            for a in others_family:
                union_f.update(algos_here[a])
            v = overlap_fraction(pts_a, union_f)
            if v is not None:
                within_vals.append(v)
        if others_cross:
            union_c = set()
            for a in others_cross:
                union_c.update(algos_here[a])
            v = overlap_fraction(pts_a, union_c)
            if v is not None:
                cross_vals.append(v)
    w = sum(within_vals) / len(within_vals) if within_vals else None
    c = sum(cross_vals) / len(cross_vals) if cross_vals else None
    return w, c


def main():
    points = load_points(RAW_RUNS_PATH)
    instances = list(points.keys())

    print("=== OMF-statique / OMF-aleatoire / OMF-RL : chevauchement entre elles ===")
    for algo in CANDIDATE_OMF_FAMILY:
        w, c = family_overlap(points, algo, CANDIDATE_OMF_FAMILY, instances)
        autres = [a for a in CANDIDATE_OMF_FAMILY if a != algo]
        if w is None:
            print(f"  {algo:15s} absent des runs -- avez-vous bien lance la campagne pour cet algo ?")
            continue
        print(f"  {algo:15s} avec {autres} = {w:.1%}   hors ce groupe = {c:.1%}")

    print("\nInterpretation : si le chevauchement intra-groupe est nettement PLUS ELEVE")
    print("que le chevauchement hors-groupe (cf. cas MOGWO/MOFA/MOABC : ~95% vs ~30%),")
    print("une fusion de famille est necessaire dans model/benchmark_runner.py::FAMILIES.")
    print("Sinon (intra du meme ordre que hors-groupe), AUCUNE fusion n'est necessaire --")
    print("chaque algo reste dans sa propre famille (deja le cas par defaut).")


if __name__ == "__main__":
    main()
