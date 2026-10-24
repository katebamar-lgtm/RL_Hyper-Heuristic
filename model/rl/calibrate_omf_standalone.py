"""
Calibration de NF, IT, NN, R et du critère de dominance pour le pilote OMF
autonome (`model/omf_standalone.py::run_omf`), selon le MÊME protocole que
le reste du dépôt (`model/calibration.py`) : budget d'évaluations fixe
(EvalCounter), instances de calibration DISTINCTES du jeu de test final
(`build_tuning_instances()`, seeds 101-104, réutilisées telles quelles,
aucune nouvelle instance créée ici), recherche séquentielle par groupe de
paramètres, sélection robuste par test de Mann-Whitney
(`select_robust_best`, réutilisée sans modification).

NF (nombre de démarrages) n'existe pas dans une architecture à
population classique (démarrages indépendants, pas génération) -- calibré en
premier ici car c'est le paramètre qui gouverne le plus directement le
compromis exploration (NF grand, IT/NN petits) vs exploitation (NF petit,
IT/NN grands) propre à ce pilote.

Point de départ des étapes 2 et 3 : la configuration DÉJÀ calibrée de
un algorithme antérieur (IT=10, NN=4, R=0.5, dominance="strict"), pour ancrer
la recherche sur une référence connue plutôt que sur des valeurs
arbitraires. Cela ne présume PAS que ces valeurs sont optimales pour ce
pilote (population vs démarrages multiples = dynamiques différentes) --
seule la calibration ci-dessous en décidera.
"""

from __future__ import annotations
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

from ..generator import generate_network
from ..pareto import normalize, hypervolume
from ..budget import EvalCounter
from ..omf_standalone import run_omf
from ..calibration import (
    TuningInstance, build_tuning_instances, build_multiscale_tuning_instances,
    _pick_od, select_robust_best, CALIB_BUDGET, CALIB_REPLICATIONS,
)
from .train_instances import ALL_MODES

NF_GRID = [3, 5, 8, 12, 20]
IT_NN_GRID = [(20, 2), (10, 4), (8, 5), (5, 8), (4, 10), (2, 20)]  # produit constant ~40
R_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
DOMINANCE_GRID = ["weak", "strict"]

# Ancrage de continuité avec une configuration de référence (cf. docstring module).
_ANCHOR_IT, _ANCHOR_NN, _ANCHOR_R, _ANCHOR_DOM = 10, 4, 0.5, "strict"


def _hv_of_objs(objs: List[Tuple[float, ...]], ref_margin: float = 1.05,
                 n_samples: int = 4000, rng_seed: int = 42) -> float:
    """Analogue de `calibration.py::_hv_of_front`, adapté : `run_omf`
    retourne directement une liste de tuples-objectifs, pas une liste
    d'individus porteurs d'un attribut `.objectives`."""
    if not objs:
        return 0.0
    pts = list(objs)
    if len(pts) == 1:
        pts = pts * 2  # normalize() a besoin d'un peu de variance ; borne dégénérée
    norm_objs, ideal, nadir = normalize(pts)
    m = len(norm_objs[0])
    ref = tuple(ref_margin for _ in range(m))
    ideal0 = tuple(0.0 for _ in range(m))
    return hypervolume(list(norm_objs), ref, ideal0, n_samples=n_samples, rng=random.Random(rng_seed))


def _evaluate(build_kwargs: Callable[[int, int], dict],
              instances: List[TuningInstance],
              n_replications: int, budget: int) -> Tuple[float, List[float]]:
    """`build_kwargs(seed, budget) -> dict` de kwargs (nf, it, nn, r,
    dominance) pour `run_omf`. Retourne (hv_moyen, valeurs_brutes)."""
    hv_values = []
    for inst in instances:
        net = generate_network(inst.n, inst.nb, inst.nt, inst.nm, seed=inst.seed)
        origin, destination = _pick_od(net)
        for rep in range(n_replications):
            seed = 1000 * inst.seed + rep
            kwargs = build_kwargs(seed, budget)
            counter = EvalCounter(budget=budget)
            _, objs = run_omf(net, origin, destination, ALL_MODES,
                               counter=counter, rng=random.Random(seed), **kwargs)
            hv_values.append(_hv_of_objs(objs))
    return sum(hv_values) / len(hv_values), hv_values


def _grid_search(name: str, grid: List[Tuple[object, Callable]],
                  instances: List[TuningInstance], n_replications: int, budget: int,
                  verbose: bool) -> List[Tuple[object, float, List[float]]]:
    results = []
    t0 = time.time()
    for i, (label, build_kwargs) in enumerate(grid):
        mean_hv, raw = _evaluate(build_kwargs, instances, n_replications, budget)
        results.append((label, mean_hv, raw))
        if verbose:
            print(f"[{name}] [{i+1}/{len(grid)}] {label} -> HV moyen={mean_hv:.4f} "
                  f"(elapsed={time.time()-t0:.0f}s)")
    results.sort(key=lambda x: -x[1])
    return results


def calibrate_omf_standalone(verbose: bool = True,
                              instances: Optional[List[TuningInstance]] = None,
                              budget: int = CALIB_BUDGET,
                              n_replications: int = CALIB_REPLICATIONS
                              ) -> Tuple[Dict, float, List, Tuple]:
    """Recherche séquentielle en 3 étapes : NF, puis (IT,NN), puis
    (R, dominance). Retourne (config, hv_moyen, ex_aequo, (r1,r2,r3))
    -- même forme de retour que les autres fonctions calibrate_*(), pour rester
    directement exploitable par un futur export JSON au même format que
    `calibration/final_*.json`.
    """
    instances = instances or build_tuning_instances()

    def _mk(nf, it, nn, r, dom):
        return lambda seed, bud: dict(nf=nf, it=it, nn=nn, r=r, dominance=dom)

    stage1 = [(nf, _mk(nf, _ANCHOR_IT, _ANCHOR_NN, _ANCHOR_R, _ANCHOR_DOM)) for nf in NF_GRID]
    r1 = _grid_search("OMF-standalone (étape 1 : NF)", stage1, instances, n_replications, budget, verbose)
    best_nf = r1[0][0]

    stage2 = [((it, nn), _mk(best_nf, it, nn, _ANCHOR_R, _ANCHOR_DOM)) for it, nn in IT_NN_GRID]
    r2 = _grid_search("OMF-standalone (étape 2 : IT,NN)", stage2, instances, n_replications, budget, verbose)
    best_it, best_nn = r2[0][0]

    stage3 = [((r, dom), _mk(best_nf, best_it, best_nn, r, dom))
              for r in R_GRID for dom in DOMINANCE_GRID]
    r3 = _grid_search("OMF-standalone (étape 3 : R,dominance)", stage3, instances, n_replications, budget, verbose)

    best_label, best_hv, ties = select_robust_best(r3)
    best_r, best_dom = best_label
    config = {"nf": best_nf, "it": best_it, "nn": best_nn, "r": best_r, "dominance": best_dom}
    return config, best_hv, ties, (r1, r2, r3)


if __name__ == "__main__":
    import json
    import os

    print("Calibration OMF-standalone -- jeu de calibration small (tune-A..D).")
    print("Réduit à n_replications=2 pour un test de fumée rapide ; relancer")
    print("avec les valeurs par défaut (CALIB_REPLICATIONS, CALIB_BUDGET) pour")
    print("la calibration définitive.\n")

    config, hv, ties, _ = calibrate_omf_standalone(n_replications=2)

    print("\n=== Résultat ===")
    print("Configuration retenue :", config)
    print("HV moyen :", round(hv, 4))
    print("Ex-aequo statistiques (Mann-Whitney, p>0.05) :", ties)

    os.makedirs("calibration", exist_ok=True)
    out_path = os.path.join("calibration", "final_omf_standalone_smoketest.json")
    with open(out_path, "w") as f:
        json.dump({"config": config, "hv": hv, "ties": [str(t) for t in ties]}, f, indent=2)
    print(f"\nÉcrit dans {out_path} "
          "(SMOKE TEST -- ne pas confondre avec une calibration définitive).")
