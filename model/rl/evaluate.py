"""
Diagnostic de généralisation -- PAS la campagne de comparaison finale
(celle-ci utilisera `model/benchmark_runner.py` + le jeu de test officiel,
cf. plan d'implémentation, étape 7). Ce module répond à une question plus
étroite et plus rapide à vérifier : la politique apprise (gelée, cf.
`FrozenPolicy`) se comporte-t-elle raisonnablement sur des instances
JAMAIS vues à l'entraînement, avant d'engager une campagne complète ?

Calcule, pour chaque condition (OMF-statique small, OMF-statique
multi-échelle, OMF-aléatoire, OMF-RL gelé) :
  - l'hypervolume normalisé sur les instances d'ENTRAÎNEMENT (mémorisation)
  - l'hypervolume normalisé sur les instances de DIAGNOSTIC (généralisation)
  - l'écart entre les deux ("écart de généralisation", protocole §6)

"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple

from ..budget import EvalCounter
from ..pareto import normalize, hypervolume
from ..omf_standalone import run_omf
from .agent import FrozenPolicy, RandomController
from .train_instances import build_rl_training_instances, ALL_MODES
from .diagnostic_instances import build_diagnostic_instances
from .training import (DEFAULT_NF, DEFAULT_NN, DEFAULT_IT, DEFAULT_DOMINANCE,
                        _BUDGET_MULTIPLIER, _POP_SIZE_BY_CATEGORY)

# Configuration calibrée sur le jeu SMALL uniquement
# (calibration/final_omf_standalone.json) -- baseline "naive", reproduit
# le problème de calibration non recalibrée à l'échelle, identifié plus tôt dans le projet.
STATIC_SMALL = dict(nf=8, it=20, nn=2, r=0.5, dominance="strict")

# Configuration calibrée multi-échelle
# (calibration/final_omf_standalone_multiscale.json) -- baseline plus
# robuste, déjà un compromis statique sur toute la gamme de tailles.
STATIC_MULTISCALE = dict(nf=20, it=20, nn=2, r=0.25, dominance="weak")


def _hv_of_objs(objs: List[Tuple[float, ...]], ref_margin: float = 1.05,
                n_samples: int = 4000, rng_seed: int = 42) -> float:
    if not objs:
        return 0.0
    pts = list(objs)
    if len(pts) == 1:
        pts = pts * 2
    norm_objs, ideal, nadir = normalize(pts)
    m = len(norm_objs[0])
    ref = tuple(ref_margin for _ in range(m))
    ideal0 = tuple(0.0 for _ in range(m))
    return hypervolume(list(norm_objs), ref, ideal0, n_samples=n_samples, rng=random.Random(rng_seed))


def _run_condition_on_instance(inst, controller, seed: int, static_kwargs: dict = None):
    net = inst.build()
    origin, destination = inst.endpoints(net)
    pop_size = _POP_SIZE_BY_CATEGORY[inst.category]
    budget = _BUDGET_MULTIPLIER * pop_size
    counter = EvalCounter(budget=budget)
    kwargs = static_kwargs or dict(nf=DEFAULT_NF, it=DEFAULT_IT, nn=DEFAULT_NN, dominance=DEFAULT_DOMINANCE)
    _, objs = run_omf(net, origin, destination, ALL_MODES,
                       counter=counter, controller=controller, rng=random.Random(seed), **kwargs)
    return objs


def generalization_report(trained_agent, seed: int = 999, verbose: bool = True) -> Dict:
    """Compare 4 conditions sur les instances d'ENTRAÎNEMENT (601-622) et
    sur les instances de DIAGNOSTIC (701/711/721, jamais vues).

    `trained_agent` : un ContextualBandit déjà entraîné (cf. train() ou
    train_multi_seed()) -- gelé ici via FrozenPolicy avant toute
    évaluation (cf. docstring de FrozenPolicy : aucun apprentissage ne
    doit se produire pendant une évaluation/comparaison).
    """
    frozen = FrozenPolicy(trained_agent)
    train_instances = build_rl_training_instances()
    diag_instances = build_diagnostic_instances()

    # OMF-RL est évalué avec la configuration (nf, nn, it, dominance) qui a
    # SERVI À L'ENTRAÎNER (`frozen.trained_config`), pas avec les valeurs
    # DEFAULT_* -- sinon un agent entraîné avec un NN non standard (ex.
    # ablation NN=8) serait évalué à NN=2, ce qui invalide la comparaison
    # (bug trouvé et corrigé lors d'une ablation manuelle, cf. README).
    rl_kwargs = frozen.trained_config  # None si l'agent n'a pas cet attribut (rétrocompat)

    conditions = {
        "OMF-statique (small)": (None, STATIC_SMALL),
        "OMF-statique (multi-échelle)": (None, STATIC_MULTISCALE),
        "OMF-aléatoire": (RandomController(rng=random.Random(seed)), None),
        "OMF-RL (gelé)": (frozen, rl_kwargs),
    }

    report: Dict[str, Dict[str, float]] = {}
    for label, (controller, static_kwargs) in conditions.items():
        train_hv = []
        for inst in train_instances:
            objs = _run_condition_on_instance(inst, controller, seed, static_kwargs)
            train_hv.append(_hv_of_objs(objs))
        diag_hv = []
        for inst in diag_instances:
            objs = _run_condition_on_instance(inst, controller, seed, static_kwargs)
            diag_hv.append(_hv_of_objs(objs))

        mean_train = sum(train_hv) / len(train_hv)
        mean_diag = sum(diag_hv) / len(diag_hv)
        gap = mean_diag - mean_train   # négatif = dégradation en généralisation
        report[label] = {"hv_train": mean_train, "hv_diag": mean_diag, "gap": gap}
        if verbose:
            print(f"  {label:32s} HV_train={mean_train:.4f}  HV_diag={mean_diag:.4f}  "
                  f"écart={gap:+.4f}")

    if verbose:
        n_unseen = getattr(frozen, "n_unseen_states", 0)
        print(f"\n  États rencontrés en diagnostic mais absents de la politique "
              f"apprise (repli 'width') : {n_unseen}")

    return report


if __name__ == "__main__":
    from .training import train

    print("Entraînement (test rapide, 200 épisodes -- augmenter pour la version définitive)...")
    agent, _ = train(n_episodes=200, seed=0, verbose=False)
    print(f"{len(agent.q)} états appris.\n")

    print("=== Rapport de généralisation (entraînement vs diagnostic) ===")
    generalization_report(agent, verbose=True)

    print("\n=== Taux d'action par tranche de progression du budget ===")
    for bucket, rates in sorted(agent.action_rate_by_progress().items()):
        print(f"  bucket {bucket} : {rates}")
