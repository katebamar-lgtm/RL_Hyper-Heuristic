"""Run the Lyon data-derived case-study campaign with paper-matched defaults.

The manuscript's final Lyon campaign contains eight algorithms on three routes:
MOEA/D, RL-MOEA/D, RL-MOEA/D-random, NSGA-II and SPEA2 use 10 repetitions;
MOGWO, MOFA and MOABC use 5 repetitions. No OMF condition was run on Lyon.
The default output is results/reproduction/lyon_raw_runs.jsonl, so the included
case_study_raw_runs.jsonl is protected. Existing (instance, algorithm, repetition)
keys are skipped in the selected output file.

Usage from the project root:
    python scripts/run_trajet.py data/lyon
    python scripts/run_trajet.py data/lyon --out results/reproduction/lyon_raw_runs.jsonl

WARNING: this script executes optimization algorithms. To regenerate the
paper's indicators/statistics without rerunning algorithms, use
scripts/postprocessing/run_full_postprocessing.py instead.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.data_loader import load_network_from_csv
from model.benchmark_runner import ALGO_RUNNERS
from model.benchmark_instances import ALL_MODES
from model.budget import EvalCounter

ALGOS = list(ALGO_RUNNERS.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--population", type=int, default=80,
                         help="taille de population (défaut 80, identique à la catégorie 'large' synthétique)")
    parser.add_argument("--budget", type=int, default=24000,
                         help="budget d'évaluations (défaut 24000 = 300 x 80, règle déjà retenue §4.2)")
    paper_algos = ["MOEA/D", "RL-MOEA/D", "RL-MOEA/D-random", "NSGA-II", "SPEA2", "MOGWO", "MOFA", "MOABC"]
    weighted_algos = {"MOGWO", "MOFA", "MOABC"}
    parser.add_argument("--reps", type=int, default=10,
                        help="repetitions for MOEA/D, RL-MOEA/D, NSGA-II and SPEA2 (default 10)")
    parser.add_argument("--weighted-reps", type=int, default=5,
                        help="repetitions for MOGWO, MOFA and MOABC (default 5, matching the paper)")
    parser.add_argument("--algos", default=",".join(paper_algos),
                        help=f"comma-separated algorithms; paper default: {paper_algos}")
    parser.add_argument("--out", default=None, help="output JSONL; default is results/reproduction/lyon_raw_runs.jsonl")
    args = parser.parse_args()
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    for a in algos:
        if a not in ALGO_RUNNERS:
            sys.exit(f"Algorithme inconnu : '{a}' -- attendu parmi {ALGOS}")

    instances_path = os.path.join(args.data_dir, "trajet.csv")
    if not os.path.exists(instances_path):
        sys.exit(f"{instances_path} introuvable.\n"
                 f"Créez-le avec les colonnes 'name,origin,destination' (3-5 trajets réels), "
                 f"cf. docstring de ce script.")
    with open(instances_path, newline="", encoding="utf-8") as f:
        instances = list(csv.DictReader(f))

    net = load_network_from_csv(args.data_dir)
    unknown = [a for a in algos if a not in ALGO_RUNNERS]
    if unknown:
        sys.exit(f"Algorithmes inconnus : {unknown} -- attendus parmi {sorted(ALGO_RUNNERS)}")

    print(net.summary())
    print(f"\n{len(instances)} trajet(s); population={args.population}, budget={args.budget}")
    print("Répetitions par algorithme:")
    for a in algos:
        n = args.weighted_reps if a in weighted_algos else args.reps
        print(f"  {a}: {n}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(project_root, "results", "reproduction", "lyon_raw_runs.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                done.add((d["instance"], d["algo"], d["rep"]))
        if done:
            print(f"Reprise : {len(done)} run(s) déjà présents dans {out_path}, non refaits.\n")

    with open(out_path, "a") as f:
        for inst in instances:
            if inst["origin"] not in net.nodes or inst["destination"] not in net.nodes:
                print(f"ATTENTION : trajet '{inst['name']}' ignoré "
                      f"('{inst['origin']}' ou '{inst['destination']}' introuvable dans nodes.csv)")
                continue
            for algo in algos:
                n_reps = args.weighted_reps if algo in weighted_algos else args.reps
                for rep in range(n_reps):
                    if (inst["name"], algo, rep) in done:
                        continue
                    seed = 1000 + rep
                    counter = EvalCounter(budget=args.budget)
                    t0 = time.time()
                    front = ALGO_RUNNERS[algo](net, inst["origin"], inst["destination"], ALL_MODES,
                                                args.population, args.budget, seed, counter)
                    elapsed = time.time() - t0
                    objs = [ind.objectives for ind in front]
                    row = dict(instance=inst["name"], algo=algo, rep=rep, seed=seed,
                               n_evals=counter.count, budget=args.budget, front_size=len(objs),
                               cpu_seconds=elapsed, objectives=objs)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"  {inst['name']:25s} | {algo:16s} | rep {rep + 1:2d}/{n_reps} | "
                          f"évals={counter.count:6d}/{args.budget} | front={len(objs):3d} | "
                          f"temps={elapsed:6.2f}s")

    print(f"\nRésultats bruts écrits dans {out_path}")
    print("Étape suivante (indicateurs du papier) : python scripts/postprocessing/recompute_indicators.py .")


if __name__ == "__main__":
    main()
