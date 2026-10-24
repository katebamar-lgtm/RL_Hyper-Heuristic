"""
Harnais de calibration -- SPEA2, MOEA/D, MOGWO, MOFA, MOABC, RL-MOEA/D
via le moteur générique `evaluate_algo_config`/`run_algo_grid_search`.

Principe (cf. discussion méthodologique) :
  - Métrique cible : hypervolume normalisé (Monte-Carlo), moyenné sur
    plusieurs runs indépendants et plusieurs instances de calibration.
  - Budget d'ÉVALUATIONS FIXE entre configurations (EvalCounter).
  - Jeu d'instances de CALIBRATION distinct du jeu de TEST final (évite le
    surapprentissage des paramètres sur les instances d'évaluation).
  - Recherche SÉQUENTIELLE (par groupe de paramètres) plutôt que grille
    factorielle complète pour les algorithmes à plusieurs hyperparamètres :
    on fixe les paramètres secondaires à une valeur par défaut raisonnable,
    on optimise le(s) paramètre(s) principal(aux) en premier, puis on fixe
    ce choix et on affine les paramètres secondaires. Compromis pragmatique
    standard quand une grille factorielle complète est trop coûteuse (cf.
    esprit de la Sequential Parameter Optimization — Bartz-Beielstein et
    al.) — limite explicite à assumer : les interactions entre paramètres
    non testées conjointement peuvent être sous-estimées.
  - Sélection finale non pas juste sur la moyenne d'hypervolume la plus
    haute, mais avec un test de Mann-Whitney contre les configurations
    proches en moyenne (`select_robust_best`) : à qualité statistiquement
    indiscernable, on préfère la configuration la plus simple/économe.

Ce module fournit :
  - build_tuning_instances() : instances de calibration (petites/moyennes,
    seeds fixes, reproductible)
  - evaluate_algo_config()/run_algo_grid_search()/select_robust_best() :
    moteur générique budget-based, réutilisé par calibrate_spea2(),
    calibrate_moead(), calibrate_mogwo(), calibrate_mofa(),
    calibrate_moabc(), calibrate_rl_moead()
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import scipy.stats as _stats

from .entities import Mode
from .generator import generate_network
from .pareto import normalize, hypervolume
from .budget import EvalCounter

ALL_MODES = {Mode.WALK, Mode.BIKE, Mode.CAR, Mode.BUS, Mode.TRAM, Mode.METRO}


@dataclass
class TuningInstance:
    name: str
    n: int
    nb: int
    nt: int
    nm: int
    seed: int


def build_tuning_instances() -> List[TuningInstance]:
    """4 instances de calibration, volontairement plus PETITES que le futur
    jeu de test final (pour que la calibration reste rapide) et couvrant
    différentes tailles/densités de lignes TC. Seeds fixées -> reproductible.
    """
    return [
        TuningInstance("tune-A", n=12, nb=1, nt=1, nm=0, seed=101),
        TuningInstance("tune-B", n=18, nb=2, nt=1, nm=1, seed=102),
        TuningInstance("tune-C", n=20, nb=1, nt=2, nm=1, seed=103),
        TuningInstance("tune-D", n=25, nb=3, nt=1, nm=1, seed=104),
    ]


def build_tuning_instances_medium_large() -> List[TuningInstance]:
    """Instances de calibration à l'échelle MEDIUM/LARGE (constat de la
    Phase 4 : les paramètres calibrés uniquement sur tune-A..D, tous <=25
    nœuds, dégradent la couverture du front -- IGD -- de NSGA-II/
    SPEA2/MOEA-D à mesure que le réseau grandit ; jamais testés à cette
    échelle jusqu'ici). Seeds >= 301, STRICTEMENT DISJOINTES des seeds de
    calibration small (101-104) ET des 9 instances de TEST FINAL
    (`model/benchmark_instances.py`, seeds 201-223) -- condition impérative
    pour ne pas contaminer la comparaison finale (mêmes données utilisées
    pour calibrer ET pour évaluer = surapprentissage, invaliderait toute la
    comparaison)."""
    return [
        TuningInstance("tune-M-A", n=40, nb=2, nt=1, nm=1, seed=301),
        TuningInstance("tune-M-B", n=48, nb=3, nt=2, nm=0, seed=302),
        TuningInstance("tune-L-A", n=85, nb=4, nt=2, nm=1, seed=311),
        TuningInstance("tune-L-B", n=95, nb=3, nt=3, nm=1, seed=312),
    ]


def build_multiscale_tuning_instances() -> List[TuningInstance]:
    """Jeu de calibration COMPLET (small + medium + large) : 8 instances au
    total. Utilisé pour la re-calibration de NSGA-II/SPEA2/MOEA-D
    (Phase 4-bis) afin d'obtenir des paramètres robustes sur toute la
    gamme de tailles évaluées, plutôt que sur-ajustés au régime "small"
    """
    return build_tuning_instances() + build_tuning_instances_medium_large()


def _pick_od(net) -> Tuple[str, str]:
    """Origine/destination = les deux nœuds les plus éloignés (L1 sur les
    coordonnées) -> garantit un trajet non trivial sur chaque instance."""
    nodes = sorted(net.nodes.values(), key=lambda nd: nd.coord.x + nd.coord.y)
    return nodes[0].id, nodes[-1].id


# ==================================================================== #
# Moteur générique budget-based -- réutilisé pour calibrer NSGA-II/
# SPEA2/MOEA-D/MOGWO/MOFA/MOABC/RL-MOEA/D.
# ==================================================================== #

CALIB_POPULATION = 30
CALIB_BUDGET = 2000           # NSGA-II/SPEA2/MOEA-D/RL-MOEA-D (~0.1-0.3s/run)
CALIB_BUDGET_CONTINUOUS = 1500  # MOGWO/MOFA/MOABC : décodage Dijkstra ~10x
                                 # plus coûteux (Phase 1) -> budget calibration
                                 # réduit pour rester praticable en temps de
                                 # calcul, DIFFÉRENT du budget=300xpop utilisé
                                 # dans la comparaison finale (Phase 4) ; à
                                 # documenter explicitement comme limite.
CALIB_REPLICATIONS = 5
CALIB_REPLICATIONS_CONTINUOUS = 3  # idem, réduit pour les 3 algos coûteux


def _hv_of_front(front, ref_margin: float = 1.05, n_samples: int = 4000,
                  rng_seed: int = 42) -> float:
    if not front:
        return 0.0
    objs = [ind.objectives for ind in front]
    if len(objs) == 1:
        objs = objs * 2  # normalize() a besoin d'un peu de variance ; borne dégénérée
    norm_objs, ideal, nadir = normalize(objs)
    ref = tuple(ref_margin for _ in range(4))
    ideal0 = tuple(0.0 for _ in range(4))
    return hypervolume(norm_objs, ref, ideal0, n_samples=n_samples, rng=random.Random(rng_seed))


def evaluate_algo_config(run_fn: Callable, build_params: Callable,
                          instances: List[TuningInstance],
                          n_replications: int, budget: int, population_size: int
                          ) -> Tuple[float, List[float]]:
    """Exécute UNE configuration (via `build_params(population_size, budget,
    seed) -> Params`) sur toutes les instances x répétitions, budget
    d'évaluations fixé par EvalCounter. Retourne (hv_moyen, valeurs_brutes)."""
    hv_values = []
    for inst in instances:
        net = generate_network(inst.n, inst.nb, inst.nt, inst.nm, seed=inst.seed)
        origin, destination = _pick_od(net)
        for rep in range(n_replications):
            seed = 1000 * inst.seed + rep
            params = build_params(population_size, budget, seed)
            counter = EvalCounter(budget=budget)
            front = run_fn(net, origin, destination, ALL_MODES, params, counter=counter)
            hv_values.append(_hv_of_front(front))
    return sum(hv_values) / len(hv_values), hv_values


def run_algo_grid_search(name: str, run_fn: Callable,
                          grid: List[Tuple[Dict, Callable]],
                          instances: Optional[List[TuningInstance]] = None,
                          n_replications: int = CALIB_REPLICATIONS,
                          budget: int = CALIB_BUDGET,
                          population_size: int = CALIB_POPULATION,
                          verbose: bool = True) -> List[Tuple[Dict, float, List[float]]]:
    """`grid` : liste de (label_config, build_params_fn). Retourne
    [(label_config, hv_moyen, valeurs_brutes), ...] trié décroissant."""
    instances = instances or build_tuning_instances()
    results = []
    t0 = time.time()
    for i, (label, build_params) in enumerate(grid):
        mean_hv, raw = evaluate_algo_config(run_fn, build_params, instances,
                                             n_replications, budget, population_size)
        results.append((label, mean_hv, raw))
        if verbose:
            print(f"[{name}] [{i+1}/{len(grid)}] {label} -> HV moyen={mean_hv:.4f} "
                  f"(elapsed={time.time()-t0:.0f}s)")
    results.sort(key=lambda x: -x[1])
    return results


def select_robust_best(results: List[Tuple[Dict, float, List[float]]],
                        alpha: float = 0.05, tie_margin: float = 0.02
                        ) -> Tuple[Dict, float, List[Dict]]:
    """Retient la config de meilleure moyenne, puis identifie via
    Mann-Whitney U (bilatéral) les configs voisines en moyenne (à
    `tie_margin` relatif près) qui en sont STATISTIQUEMENT INDISCERNABLES
    (p > alpha) — pas de meilleur choix unique parmi elles sans plus de
    répétitions, à signaler comme tel dans l'article plutôt que de trancher
    arbitrairement sur la 3e décimale d'une moyenne bruitée.
    Retourne (meilleure_config, son_hv_moyen, liste_des_configs_ex-aequo)."""
    best_label, best_hv, best_raw = results[0]
    ties = [best_label]
    for label, hv, raw in results[1:]:
        if best_hv <= 0 or hv < best_hv * (1 - tie_margin):
            break
        try:
            _, p = _stats.mannwhitneyu(best_raw, raw, alternative="two-sided")
        except ValueError:
            p = 1.0  # échantillons identiques (variance nulle des deux côtés)
        if p > alpha:
            ties.append(label)
    return best_label, best_hv, ties


# ==================================================================== #
# Calibration par algorithme
# ==================================================================== #

def calibrate_nsga2(verbose: bool = True, instances: Optional[List[TuningInstance]] = None):
    """NSGA-II n'avait jamais été calibré (valeurs par défaut de la
    littérature utilisées telles quelles) — corrigé ici pour que les 4
    algos à encodage discret soient calibrés selon le MÊME protocole avant
    la comparaison finale. Recherche séquentielle : crossover_rate d'abord
    (mutation=0.2 fixé), puis mutation_rate avec le meilleur crossover
    retenu."""
    from .nsga2 import NSGA2Params, run_nsga2 as _run

    def _mk(cx, mut):
        return lambda pop, budget, seed: NSGA2Params(
            population_size=pop, generations=10**9, budget=budget, seed=seed,
            crossover_rate=cx, mutation_rate=mut)

    cx_grid = [0.7, 0.8, 0.9]
    stage1 = [(cx, _mk(cx, 0.2)) for cx in cx_grid]
    r1 = run_algo_grid_search("NSGA-II (étape 1: crossover)", _run, stage1, instances=instances, verbose=verbose)
    best_cx, _, _ = r1[0]

    mut_grid = [0.1, 0.2, 0.3]
    stage2 = [(mut, _mk(best_cx, mut)) for mut in mut_grid]
    r2 = run_algo_grid_search("NSGA-II (étape 2: mutation)", _run, stage2, instances=instances, verbose=verbose)

    best_mut, best_hv, ties = select_robust_best(r2)
    config = {"crossover_rate": best_cx, "mutation_rate": best_mut}
    return config, best_hv, ties, (r1, r2)


def calibrate_spea2(verbose: bool = True, instances: Optional[List[TuningInstance]] = None):
    """Recherche séquentielle : (crossover_rate, mutation_rate) d'abord
    (archive_ratio=0.5 fixé), puis archive_ratio avec le meilleur couple
    retenu."""
    from .spea2 import SPEA2Params, run_spea2 as _run

    def _mk(cx, mut, arc_ratio):
        return lambda pop, budget, seed: SPEA2Params(
            population_size=pop, archive_size=max(2, round(pop * arc_ratio)),
            generations=10**9, budget=budget, seed=seed,
            crossover_rate=cx, mutation_rate=mut)

    cx_grid = [0.7, 0.8, 0.9]
    mut_grid = [0.1, 0.2, 0.3]
    stage1 = [((cx, mut), _mk(cx, mut, 0.5)) for cx in cx_grid for mut in mut_grid]
    r1 = run_algo_grid_search("SPEA2 (étape 1: cx,mut)", _run, stage1, instances=instances, verbose=verbose)
    (best_cx, best_mut), _, _ = r1[0]

    arc_grid = [0.3, 0.5, 0.7]
    stage2 = [(arc, _mk(best_cx, best_mut, arc)) for arc in arc_grid]
    r2 = run_algo_grid_search("SPEA2 (étape 2: archive_ratio)", _run, stage2, instances=instances, verbose=verbose)

    best_arc, best_hv, ties = select_robust_best(r2)
    config = {"crossover_rate": best_cx, "mutation_rate": best_mut, "archive_ratio": best_arc}
    return config, best_hv, ties, (r1, r2)


def calibrate_moead(verbose: bool = True, instances: Optional[List[TuningInstance]] = None):
    """Recherche séquentielle : neighborhood_size (T) d'abord
    (crossover/mutation par défaut NSGA-II), puis crossover x mutation avec
    le meilleur T retenu."""
    from .moead import MOEADParams, run_moead as _run

    def _mk(t, cx, mut):
        return lambda pop, budget, seed: MOEADParams(
            population_size=pop, generations=10**9, budget=budget, seed=seed,
            neighborhood_size=t, crossover_rate=cx, mutation_rate=mut)

    t_grid = [5, 8, 12, 16]
    stage1 = [(t, _mk(t, 0.9, 0.2)) for t in t_grid]
    r1 = run_algo_grid_search("MOEA/D (étape 1: T)", _run, stage1, instances=instances, verbose=verbose)
    best_t, _, _ = r1[0]

    cx_grid, mut_grid = [0.7, 0.9], [0.1, 0.2, 0.3]
    stage2 = [((cx, mut), _mk(best_t, cx, mut)) for cx in cx_grid for mut in mut_grid]
    r2 = run_algo_grid_search("MOEA/D (étape 2: cx,mut)", _run, stage2, instances=instances, verbose=verbose)

    (best_cx, best_mut), best_hv, ties = select_robust_best(r2)
    config = {"neighborhood_size": best_t, "crossover_rate": best_cx, "mutation_rate": best_mut}
    return config, best_hv, ties, (r1, r2)


def calibrate_mogwo(verbose: bool = True):
    """MOGWO n'a quasiment pas de paramètre libre dans le schéma GWO
    original (avantage documenté dans la littérature,
    — seul `mutation_rate` (palliatif à la
    convergence prématurée, cf. model/mogwo.py) est calibré ici. Budget/
    répétitions réduits : décodage Dijkstra ~10x plus coûteux (Phase 1)."""
    from .mogwo import MOGWOParams, run_mogwo as _run

    def _mk(mut):
        return lambda pop, budget, seed: MOGWOParams(
            population_size=pop, archive_size=pop, generations=10**9,
            budget=budget, seed=seed, mutation_rate=mut)

    grid = [(mut, _mk(mut)) for mut in [0.0, 0.1, 0.15, 0.25, 0.4]]
    r = run_algo_grid_search("MOGWO (mutation_rate)", _run, grid,
                              n_replications=CALIB_REPLICATIONS_CONTINUOUS,
                              budget=CALIB_BUDGET_CONTINUOUS, verbose=verbose)
    best_mut, best_hv, ties = select_robust_best(r)
    return {"mutation_rate": best_mut}, best_hv, ties, (r,)


def calibrate_mofa(verbose: bool = True):
    """Recherche séquentielle : (beta0, gamma) d'abord (alpha0=0.3 fixé),
    puis alpha0 avec le meilleur couple retenu. alpha_decay laissé à sa
    valeur par défaut (0.97) -- 4e paramètre libre, non calibré ici pour
    contenir le coût de calcul (limite à documenter)."""
    from .mofa import MOFAParams, run_mofa as _run

    def _mk(beta0, gamma, alpha0):
        return lambda pop, budget, seed: MOFAParams(
            population_size=pop, archive_size=pop, generations=10**9,
            budget=budget, seed=seed, beta0=beta0, gamma=gamma, alpha0=alpha0)

    beta_grid, gamma_grid = [0.5, 1.0, 1.5], [0.5, 1.0, 2.0]
    stage1 = [((b, g), _mk(b, g, 0.3)) for b in beta_grid for g in gamma_grid]
    r1 = run_algo_grid_search("MOFA (étape 1: beta0,gamma)", _run, stage1,
                               n_replications=CALIB_REPLICATIONS_CONTINUOUS,
                               budget=CALIB_BUDGET_CONTINUOUS, verbose=verbose)
    (best_beta, best_gamma), _, _ = r1[0]

    alpha_grid = [0.1, 0.3, 0.5]
    stage2 = [(a, _mk(best_beta, best_gamma, a)) for a in alpha_grid]
    r2 = run_algo_grid_search("MOFA (étape 2: alpha0)", _run, stage2,
                               n_replications=CALIB_REPLICATIONS_CONTINUOUS,
                               budget=CALIB_BUDGET_CONTINUOUS, verbose=verbose)

    best_alpha, best_hv, ties = select_robust_best(r2)
    config = {"beta0": best_beta, "gamma": best_gamma, "alpha0": best_alpha}
    return config, best_hv, ties, (r1, r2)


def calibrate_moabc(verbose: bool = True):
    """Seul hyperparamètre libre du schéma ABC standard retenu ici : `limit`
    (essais consécutifs sans amélioration avant remplacement scout)."""
    from .moabc import MOABCParams, run_moabc as _run

    def _mk(limit):
        return lambda pop, budget, seed: MOABCParams(
            population_size=pop, archive_size=pop, generations=10**9,
            budget=budget, seed=seed, limit=limit)

    grid = [(lim, _mk(lim)) for lim in [5, 10, 15, 20]]
    r = run_algo_grid_search("MOABC (limit)", _run, grid,
                              n_replications=CALIB_REPLICATIONS_CONTINUOUS,
                              budget=CALIB_BUDGET_CONTINUOUS, verbose=verbose)
    best_limit, best_hv, ties = select_robust_best(r)
    return {"limit": best_limit}, best_hv, ties, (r,)


def calibrate_rl_moead(verbose: bool = True, instances: Optional[List[TuningInstance]] = None,
                        reward_mode: str = "binary"):
    """RL-MOEA/D : recherche séquentielle, même protocole que calibrate_moead
    ci-dessus (neighborhood_size T d'abord, sur le comportement par défaut
    du bandit), puis ucb_c (coefficient d'exploration UCB1) avec le meilleur
    T retenu. neighborhood_size est calibré séparément de ucb_c car il
    affecte la taille du voisinage/contexte de récompense, pas la politique
    d'exploration elle-même -- interaction jugée secondaire (même
    hypothèse simplificatrice que pour les autres algos ci-dessus).

    `reward_mode` ("binary" | "rich") : DOIT être calibré séparément pour
    chaque mode -- l'échelle de la récompense change radicalement (binaire
    dans [0,1] vs magnitude Tchebycheff typiquement dans [0, 0.02]), donc
    la grille de `ucb_c` calibrée pour un mode n'a AUCUNE validité pour
    l'autre : le terme d'exploration UCB1 (`ucb_c * sqrt(log(t)/n)`) doit
    rester du même ordre de grandeur que les écarts de valeur réels entre
    bras pour que l'exploitation ait un sens, sans quoi le bandit explore
    presque uniformément quel que soit ce qu'il a appris.
    """
    from .rl_moead import RLMOEADParams, run_rl_moead as _run

    def _mk(t, c):
        return lambda pop, budget, seed: RLMOEADParams(
            population_size=pop, generations=10**9, budget=budget, seed=seed,
            neighborhood_size=t, ucb_c=c, reward_mode=reward_mode)

    # Grille de T inchangée (n'affecte pas l'échelle de la récompense).
    t_grid = [5, 8, 12, 16]
    # Grille de ucb_c dépendante du mode -- cf. avertissement ci-dessus.
    # "rich" : grille centrée sur l'ordre de grandeur observé empiriquement
    # des récompenses (magnitude Tchebycheff typiquement <= 0.02-0.03).
    c_grid = [0.5, 1.0, 1.4, 2.0, 3.0] if reward_mode == "binary" else \
             [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    anchor_c = 1.4 if reward_mode == "binary" else 0.01

    stage1 = [(t, _mk(t, anchor_c)) for t in t_grid]
    r1 = run_algo_grid_search(f"RL-MOEA/D [{reward_mode}] (étape 1: T)", _run, stage1,
                               instances=instances, verbose=verbose)
    best_t, _, _ = r1[0]

    stage2 = [(c, _mk(best_t, c)) for c in c_grid]
    r2 = run_algo_grid_search(f"RL-MOEA/D [{reward_mode}] (étape 2: ucb_c)", _run, stage2,
                               instances=instances, verbose=verbose)

    best_c, best_hv, ties = select_robust_best(r2)
    config = {"neighborhood_size": best_t, "ucb_c": best_c, "reward_mode": reward_mode}
    return config, best_hv, ties, (r1, r2)
