"""Run one synthetic instance batch for the published 11-condition campaign.

The included final data contain nine instances: 20 repetitions for small and
medium instances and 10 for large instances, for each of 11 algorithm
conditions. This script appends raw runs and skips keys already present.

Usage:
    python examples/run_phase4_batch.py final-S1 20
    python examples/run_phase4_batch.py final-L1 10 "MOEA/D,RL-MOEA/D"
    python examples/run_phase4_batch.py final-S1 20

"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.benchmark_instances import build_final_instances, ALL_MODES
from model.benchmark_runner import ALGO_RUNNERS
from model.budget import EvalCounter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_PATH = os.path.join(RESULTS_DIR, "reproduction", "phase4_raw_runs.jsonl")

ALL_ALGOS = ["NSGA-II", "SPEA2", "MOEA/D", "MOGWO", "MOFA", "MOABC",
             "RL-MOEA/D", "RL-MOEA/D-random",
             "OMF-statique", "OMF-aleatoire", "OMF-RL"]


def already_done(out_path: str) -> set:
    """(instance, algo, rep) déjà présents dans le fichier -- permet de
    relancer ce script en toute sécurité sans dupliquer des runs déjà
    calculés (reprise après un chunk précédent)."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                done.add((d["instance"], d["algo"], d["rep"]))
    return done


def run_batch(instance_name: str, n_reps: int, algos, base_seed: int = 1000, out_path: str = OUT_PATH):
    instances = {i.name: i for i in build_final_instances()}
    inst = instances[instance_name]
    net = inst.build()
    origin, destination = inst.endpoints(net)
    unknown = [a for a in algos if a not in ALGO_RUNNERS]
    if unknown:
        raise ValueError(f"Unknown algorithms {unknown}; available: {sorted(ALGO_RUNNERS)}")
    done = already_done(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as f:
        for algo in algos:
            for rep in range(n_reps):
                if (inst.name, algo, rep) in done:
                    continue
                seed = base_seed + rep
                counter = EvalCounter(budget=inst.budget)
                t0 = time.time()
                front = ALGO_RUNNERS[algo](net, origin, destination, ALL_MODES,
                                            inst.population_size, inst.budget, seed, counter)
                elapsed = time.time() - t0
                objs = [ind.objectives for ind in front]
                row = dict(algo=algo, instance=inst.name, category=inst.category,
                           rep=rep, seed=seed, n_evals=counter.count, budget=inst.budget,
                           front_size=len(objs), cpu_seconds=elapsed, objectives=objs)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  {inst.name:10s} | {algo:8s} | rep {rep+1:2d}/{n_reps} | "
                      f"évals={counter.count:6d}/{inst.budget} | front={len(objs):3d} | "
                      f"temps={elapsed:6.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_name", choices=[i.name for i in build_final_instances()])
    parser.add_argument("n_reps", type=int)
    parser.add_argument("algorithms", nargs="?", default=",".join(ALL_ALGOS),
                        help="comma-separated algorithms; default is all 11 paper conditions")
    parser.add_argument("--out", default=OUT_PATH, help=f"output JSONL (default: {OUT_PATH})")
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    run_batch(args.instance_name, args.n_reps, algos, base_seed=args.base_seed, out_path=args.out)
