"""
Ce module fixe le protocole (interface commune, front de référence,
indicateurs) ; il NE LANCE PAS la comparaison finale complète (Phase 4,
volume de calcul bien plus important, en particulier pour MOGWO/MOFA/MOABC
dont le décodage est ~10-60x plus lent.
Utiliser `examples/run_benchmark_quick.py` pour une validation rapide du
harnais avant de lancer la campagne complète.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence

from .network import MultimodalNetwork
from .benchmark_instances import BenchmarkInstance, ALL_MODES
from .budget import EvalCounter
from .pareto import (non_dominated_indices, hypervolume, normalize, bounds,
                      inverted_generational_distance, generational_distance)

from .nsga2 import NSGA2Params, run_nsga2
from .spea2 import SPEA2Params, run_spea2
from .moead import MOEADParams, run_moead
from .mogwo import MOGWOParams, run_mogwo
from .mofa import MOFAParams, run_mofa
from .moabc import MOABCParams, run_moabc
from .rl_moead import RLMOEADParams, run_rl_moead, run_rl_moead_random
from .omf_standalone import run_omf
from .moead import Individual  # réutilisé comme format de sortie commun (path, objectives)
from .rl.agent import RandomController

# Familles méthodologiques -- cf. audit du 23/08/2026 : MOGWO/MOFA/MOABC
# partagent TOUS LES TROIS le même décodage exact par Dijkstra pondéré
# (model/scenario.py::weighted_shortest_path, Phase 1). Vérifié empiriquement
# sur les 9 instances finales : 95.1% des points de ces trois algos sont
# retrouvés chez au moins un des deux autres membres de la famille, contre
# seulement 8.8% chez un algo à encodage discret -- et les 4 algos à
# encodage discret ne partagent que 9.6% entre eux (chevauchement normal).
# Un simple leave-one-out (exclure seulement l'algo évalué) sous-estime donc
# fortement le GD/IGD réel de MOGWO/MOFA/MOABC : leur référence contient
# encore leurs deux "cousins" qui redécouvrent trivialement les mêmes
# points via le même décodeur. Le leave-FAMILY-out ci-dessous exclut les
# TROIS quand on évalue l'un des trois ; pour les 4 algos discrets, la
# famille est réduite à l'algo seul (comportement inchangé, équivalent à
# l'ancien leave-one-out).
FAMILIES = {
    "NSGA-II": {"NSGA-II"},
    "SPEA2": {"SPEA2"},
    "MOEA/D": {"MOEA/D"},
    "MOGWO": {"MOGWO", "MOFA", "MOABC"},
    "MOFA": {"MOGWO", "MOFA", "MOABC"},
    "MOABC": {"MOGWO", "MOFA", "MOABC"},
    # RL-MOEA/D et sa variante d'ablation (tirage aléatoire d'opérateur)
    # partagent le MÊME décodage/opérateurs que MOEA/D (ga_common), mais
    # VÉRIFIÉ EMPIRIQUEMENT (campagne complète, 9 instances) : chevauchement
    # de points avec MOEA/D + l'autre variante = 7.6-13.5%, contre 11.6-17.0%
    # avec le reste des algorithmes -- chevauchement intra-groupe PLUS FAIBLE
    # que hors-groupe, l'inverse de ce qui justifierait une fusion de famille
    # (comparer au cas MOGWO/MOFA/MOABC ci-dessus : 90.8-99.1% intra-famille
    # contre 29.6-31.5% hors-famille -- un écart sans ambiguïté). Aucune
    # fusion nécessaire : chaque algorithme reste dans sa propre famille.
    "RL-MOEA/D": {"RL-MOEA/D"},
    "RL-MOEA/D-random": {"RL-MOEA/D-random"},
    # OMF-statique/aléatoire/RL partagent les mêmes opérateurs de voisinage
    # (largeur/profondeur, model/omf.py) entre eux -- VÉRIFIÉ EMPIRIQUEMENT
    # (campagne complète, 9 instances) : chevauchement intra-groupe
    # (2.5-11.0%) du même ordre que le chevauchement hors-groupe (1.0-7.0%),
    # pas d'élévation marquée comme pour MOGWO/MOFA/MOABC -- aucune fusion
    # nécessaire, chaque algorithme reste dans sa propre famille.
    "OMF-statique": {"OMF-statique"},
    "OMF-aleatoire": {"OMF-aleatoire"},
    "OMF-RL": {"OMF-RL"},
}

# Paramètres calibrés en Phase 4-bis (cf. calibration/final_*_multiscale.json,
# obtenus via model.calibration.calibrate_*(instances=build_multiscale_tuning_instances())
# -- 8 instances small+medium+large, budget d'évaluations fixe, recherche
# séquentielle, sélection robuste par test de Mann-Whitney). Remplace la
# calibration initiale de la Phase 3 après constat d'une
# dégradation d'IGD à l'échelle pour les algos à encodage discret. NSGA-II
# retombe sur les MÊMES valeurs qu'en calibration small-only
# SPEA2 et MOEA-D changent.
_CALIBRATED = {
    "NSGA-II": dict(crossover_rate=0.9, mutation_rate=0.2),
    "SPEA2":   dict(crossover_rate=0.7, mutation_rate=0.1, archive_ratio=0.5),
    "MOEA/D":  dict(neighborhood_size=8, crossover_rate=0.9, mutation_rate=0.2),
    "MOGWO":   dict(mutation_rate=0.15),  # cf. calibration/final_mogwo.json (choix conservateur documenté) -- inchangé
    "MOFA":    dict(beta0=1.5, gamma=1.0, alpha0=0.5),                        # inchangé
    "MOABC":   dict(limit=5),  # cf. calibration/final_moabc.json : effet non mesurable à cette échelle -- inchangé
    # RL-MOEA/D : valeurs CALIBRÉES (cf. calibration/final_rl_moead.json,
    # obtenu via run_calib_rl_moea.py / model/calibration.py::calibrate_rl_moead()
    # sur le jeu de calibration small, seeds 101-104). Remplace la valeur de
    # départ (neighborhood_size=8, ucb_c=1.4) utilisée avant calibration --
    # CORRECTION : cette config n'avait pas été mise à jour après la
    # calibration, la campagne aurait tourné avec une configuration non
    # calibrée sans avertissement.
    # RL-MOEA/D : RECALIBRE pour reward_mode="rich" (magnitude Tchebycheff,
    # cf. model/rl_moead.py) -- l'echelle de la recompense a change
    # radicalement (binaire [0,1] -> magnitude typiquement [0, 0.02]),
    # ucb_c doit etre recalibre en consequence (ancienne valeur 1.4 aurait
    # rendu l'exploration UCB1 totalement dominante, cf. discussion).
    # cf. calibration/final_rl_moead_rich.json.
    "RL-MOEA/D": dict(neighborhood_size=12, ucb_c=0.02, reward_mode="rich"),
    "RL-MOEA/D-random": dict(neighborhood_size=16),
    # OMF-statique : calibration multi-échelle (final_omf_standalone_multiscale.json),
    # SAUF nn=8 (au lieu de 2) -- diagnostic de balayage NN, jamais recalibré
    # à part entière avec ce nn (une seule campagne, pas de nouveau cycle
    # d'itération, cf. discussion) -- limite à noter explicitement dans
    # l'article, pas une calibration complète comme les autres algos.
    "OMF-statique": dict(nf=20, it=20, nn=8, r=0.25, dominance="weak"),
    "OMF-aleatoire": dict(nf=20, it=20, nn=8, dominance="weak"),
    "OMF-RL": dict(nf=20, it=20, nn=8, dominance="weak"),
}

ALGO_RUNNERS: Dict[str, Callable] = {
    "NSGA-II": lambda net, o, d, m, pop, budget, seed, counter: run_nsga2(
        net, o, d, m, NSGA2Params(population_size=pop, generations=10**9, budget=budget, seed=seed,
                                   **_CALIBRATED["NSGA-II"]), counter=counter),
    "SPEA2":   lambda net, o, d, m, pop, budget, seed, counter: run_spea2(
        net, o, d, m, SPEA2Params(population_size=pop, generations=10**9, budget=budget, seed=seed,
                                   archive_size=max(2, round(pop * _CALIBRATED["SPEA2"]["archive_ratio"])),
                                   crossover_rate=_CALIBRATED["SPEA2"]["crossover_rate"],
                                   mutation_rate=_CALIBRATED["SPEA2"]["mutation_rate"]), counter=counter),
    "MOEA/D":  lambda net, o, d, m, pop, budget, seed, counter: run_moead(
        net, o, d, m, MOEADParams(population_size=pop, generations=10**9, budget=budget, seed=seed,
                                   **_CALIBRATED["MOEA/D"]), counter=counter),
    "MOGWO":   lambda net, o, d, m, pop, budget, seed, counter: run_mogwo(
        net, o, d, m, MOGWOParams(population_size=pop, archive_size=pop, generations=10**9, budget=budget, seed=seed,
                                   **_CALIBRATED["MOGWO"]), counter=counter),
    "MOFA":    lambda net, o, d, m, pop, budget, seed, counter: run_mofa(
        net, o, d, m, MOFAParams(population_size=pop, archive_size=pop, generations=10**9, budget=budget, seed=seed,
                                  **_CALIBRATED["MOFA"]), counter=counter),
    "MOABC":   lambda net, o, d, m, pop, budget, seed, counter: run_moabc(
        net, o, d, m, MOABCParams(population_size=pop, archive_size=pop, generations=10**9, budget=budget, seed=seed,
                                   **_CALIBRATED["MOABC"]), counter=counter),
    "RL-MOEA/D": lambda net, o, d, m, pop, budget, seed, counter: run_rl_moead(
        net, o, d, m, RLMOEADParams(population_size=pop, generations=10**9, budget=budget, seed=seed,
                                     **_CALIBRATED["RL-MOEA/D"]), counter=counter),
    "RL-MOEA/D-random": lambda net, o, d, m, pop, budget, seed, counter: run_rl_moead_random(
        net, o, d, m, RLMOEADParams(population_size=pop, generations=10**9, budget=budget, seed=seed,
                                     **_CALIBRATED["RL-MOEA/D-random"]), counter=counter),
}


def _omf_result_to_individuals(paths, objs):
    """Adapte la sortie de run_omf() (paths, objs) au format commun
    Individual(path, objectives) attendu par le reste du pipeline."""
    return [Individual(p, tuple(o)) for p, o in zip(paths, objs)]


import random as _random  # local, evite tout conflit avec un eventuel `random` deja importe ailleurs

_OMF_RL_POLICY_PATH = "calibration/final_omf_rl_policy.pkl"
_omf_rl_policy_cache = None


def _load_omf_rl_policy():
    """Chargement paresseux (une seule fois) de la politique gelée --
    produite par train_final_omf_rl.py. Erreur explicite si absente,
    plutôt qu'un échec silencieux ou un import qui casse tout le module."""
    global _omf_rl_policy_cache
    if _omf_rl_policy_cache is None:
        import pickle
        try:
            with open(_OMF_RL_POLICY_PATH, "rb") as f:
                _omf_rl_policy_cache = pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"Politique OMF-RL introuvable ({_OMF_RL_POLICY_PATH}). "
                "Lancez d'abord `python train_final_omf_rl.py`."
            )
    return _omf_rl_policy_cache


ALGO_RUNNERS["OMF-statique"] = lambda net, o, d, m, pop, budget, seed, counter: _omf_result_to_individuals(
    *run_omf(net, o, d, m, counter=counter, rng=_random.Random(seed), **_CALIBRATED["OMF-statique"]))

ALGO_RUNNERS["OMF-aleatoire"] = lambda net, o, d, m, pop, budget, seed, counter: _omf_result_to_individuals(
    *run_omf(net, o, d, m, counter=counter, rng=_random.Random(seed),
             controller=RandomController(rng=_random.Random(seed)), **_CALIBRATED["OMF-aleatoire"]))

ALGO_RUNNERS["OMF-RL"] = lambda net, o, d, m, pop, budget, seed, counter: _omf_result_to_individuals(
    *run_omf(net, o, d, m, counter=counter, rng=_random.Random(seed),
             controller=_load_omf_rl_policy(), **_CALIBRATED["OMF-RL"]))


@dataclass
class RunResult:
    algo: str
    instance: str
    category: str
    rep: int
    seed: int
    n_evals: int
    budget: int
    front_size: int
    cpu_seconds: float  # Historical field name; measured with time.time() = elapsed wall-clock seconds.
    objectives: List[tuple]  # front brut (pour reconstruire le front de référence ensuite)


def run_one(algo: str, instance: BenchmarkInstance, net: MultimodalNetwork,
            origin: str, destination: str, rep: int, base_seed: int) -> RunResult:
    """Exécute UN algorithme sur UNE instance pour UNE répétition. `rep`
    change la seed (base_seed + rep) pour garantir des runs indépendants
    reproductibles."""
    seed = base_seed + rep
    runner = ALGO_RUNNERS[algo]
    counter = EvalCounter(budget=instance.budget)
    t0 = time.time()
    front = runner(net, origin, destination, ALL_MODES, instance.population_size,
                    instance.budget, seed, counter)
    elapsed = time.time() - t0
    objs = [ind.objectives for ind in front]
    return RunResult(algo, instance.name, instance.category, rep, seed,
                      counter.count, instance.budget, len(objs), elapsed, objs)


def run_instance(instance: BenchmarkInstance, algos: Sequence[str], n_reps: int,
                  base_seed: int = 1000, verbose: bool = True) -> List[RunResult]:
    net = instance.build()
    origin, destination = instance.endpoints(net)
    results: List[RunResult] = []
    for algo in algos:
        for rep in range(n_reps):
            r = run_one(algo, instance, net, origin, destination, rep, base_seed)
            results.append(r)
            if verbose:
                print(f"  {instance.name:10s} | {algo:8s} | rep {rep+1:2d}/{n_reps} | "
                      f"évals={r.n_evals:6d}/{r.budget} | front={r.front_size:3d} | "
                      f"temps={r.cpu_seconds:6.1f}s")
    return results


def reference_front(results: List[RunResult], instance_name: str,
                     exclude_algo: Optional[str] = None) -> List[tuple]:
    """
    `exclude_algo` (leave-FAMILY-out) : exclut du pool les runs de TOUTE LA
    FAMILLE méthodologique de l'algo passé en paramètre (cf. `FAMILIES`),
    pas seulement cet algo lui-même. Nécessaire pour que le GD/IGD d'un
    algo ne soit pas calculé contre une référence qui contient ses propres
    points OU ceux d'un "cousin" partageant le même mécanisme de décodage
    -- sinon un algo (ou toute sa famille) qui domine largement les autres
    se retrouve avec (quasi) tous ses points DANS la référence, donc un
    GD artificiellement proche de 0 (biais constaté sur MOABC/MOFA/MOGWO :
    ils partagent tous les trois le décodage exact par Dijkstra pondéré,
    cf. FAMILIES ci-dessus -- un simple leave-one-out ne suffisait pas,
    audit du 23/08/2026).
    `compute_indicators` appelle cette fonction une fois par algo avec
    `exclude_algo=<cet algo>` ; les autres appelants (ex. inspection
    manuelle du front global) peuvent laisser `exclude_algo=None` pour
    retrouver l'ancien comportement (union stricte, sans exclusion)."""
    excluded_family = FAMILIES.get(exclude_algo, {exclude_algo}) if exclude_algo else set()
    pool: List[tuple] = []
    for r in results:
        if r.instance == instance_name and r.algo not in excluded_family:
            pool.extend(r.objectives)
    if not pool:
        return []
    seen = set()
    dedup = []
    for p in pool:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    nd_idx = non_dominated_indices(dedup)
    return [dedup[i] for i in nd_idx]


def compute_indicators(results: List[RunResult]) -> List[dict]:
    """Ajoute HV normalisé + IGD + GD à chaque RunResult, contre un front
    de référence leave-one-out (union des non-dominés de TOUS les algos
    SAUF celui évalué, cf. `reference_front`). Un cache par
    (instance, algo évalué) évite de reconstruire la référence à chaque
    run/répétition. Retourne une liste de dicts prêts à sérialiser
    (JSONL/CSV)."""
    rows = []
    by_instance_algo: Dict[tuple, List[tuple]] = {}
    instance_names = {r.instance for r in results}
    algo_names = {r.algo for r in results}
    for name in instance_names:
        for algo in algo_names:
            by_instance_algo[(name, algo)] = reference_front(results, name, exclude_algo=algo)

    for r in results:
        ref = by_instance_algo[(r.instance, r.algo)]
        row = asdict(r)
        row.pop("objectives")  # trop volumineux pour un export agrégé ; dispo via reference_front si besoin
        if not ref or not r.objectives:
            row.update(hv=None, igd=None, gd=None)
            rows.append(row)
            continue
        union = ref + r.objectives
        ideal, nadir = bounds(union)
        cand_norm, _, _ = normalize(r.objectives, ideal, nadir)
        ref_norm, _, _ = normalize(ref, ideal, nadir)
        rp = tuple([1.1] * len(ideal))
        hv = hypervolume(cand_norm, rp, tuple([0.0] * len(ideal)), n_samples=20000)
        igd = inverted_generational_distance(r.objectives, ref)
        gd = generational_distance(r.objectives, ref)
        row.update(hv=hv, igd=igd, gd=gd)
        rows.append(row)
    return rows


def run_benchmark(instances: Sequence[BenchmarkInstance], algos: Sequence[str],
                   n_reps: int, out_path: Optional[str] = None,
                   base_seed: int = 1000, verbose: bool = True) -> List[dict]:
    """Point d'entrée complet : boucle instances x algos x répétitions,
    calcule les indicateurs, écrit un JSONL si `out_path` est fourni (une
    ligne par run, exploitable directement par le module d'analyse
    statistique de la Phase 5)."""
    all_results: List[RunResult] = []
    for instance in instances:
        if verbose:
            print(f"=== {instance.name} (n={instance.n}, budget={instance.budget}) ===")
        all_results.extend(run_instance(instance, algos, n_reps, base_seed, verbose))

    rows = compute_indicators(all_results)
    if out_path:
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return rows
