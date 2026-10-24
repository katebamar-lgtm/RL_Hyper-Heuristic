"""
Campagne d'entraînement RL "réelle" (plusieurs graines indépendantes,
davantage d'épisodes que le test de fumée) -- PAS la campagne de
comparaison finale (celle-ci utilisera `model/benchmark_runner.py` + le
jeu de test officiel + Lyon, cf. plan d'implémentation, étape 7). Ce
module répond aux questions de développement restantes avant d'y investir :

  - La politique apprise est-elle stable d'une graine d'entraînement à
    l'autre (`policy_agreement`, cf. protocole expérimental §7 point 5) ?
  - L'écart de généralisation (entraînement vs diagnostic) est-il
    cohérent d'une graine à l'autre, ou très variable (auquel cas
    l'entraînement n'a pas convergé) ?
"""

from __future__ import annotations
import statistics
from typing import List, Tuple

from .training import train_multi_seed, policy_agreement, ContextualBandit
from .agent import FrozenPolicy
from .evaluate import generalization_report, _run_condition_on_instance, _hv_of_objs
from .train_instances import build_rl_training_instances
from .diagnostic_instances import build_diagnostic_instances


def run_campaign(n_seeds: int = 5, n_episodes: int = 1000, base_seed: int = 0,
                  verbose: bool = True
                  ) -> Tuple[List[Tuple[int, ContextualBandit, List[dict]]], list, tuple]:
    print(f"=== Entraînement de {n_seeds} agents indépendants ({n_episodes} épisodes chacun) ===")
    agents = train_multi_seed(n_seeds=n_seeds, n_episodes=n_episodes,
                               base_seed=base_seed, verbose=verbose)

    print("\n=== Accord de politique inter-graines (états partagés) ===")
    agreements = policy_agreement(agents)
    for s1, s2, rate, n_shared in agreements:
        if rate is None:
            print(f"  graine {s1} vs {s2} : aucun état partagé")
        else:
            print(f"  graine {s1} vs {s2} : accord={rate:.2%} sur {n_shared} états partagés")
    rates = [r for _, _, r, _ in agreements if r is not None]
    if rates:
        print(f"  -> accord moyen : {statistics.mean(rates):.2%}")

    print("\n=== Rapport de généralisation PAR GRAINE (OMF-RL uniquement -- les baselines")
    print("    statiques/aléatoire ne dépendent pas de la graine, cf. generalization_report) ===")
    train_hvs, diag_hvs, gaps = [], [], []
    for seed, agent, _ in agents:
        frozen = FrozenPolicy(agent)
        rl_kwargs = frozen.trained_config   # cf. correction dans evaluate.py -- évalue
                                              # avec la config REELLEMENT entraînée, pas DEFAULT_*
        train_objs_hv = [
            _hv_of_objs(_run_condition_on_instance(inst, frozen, 999, rl_kwargs))
            for inst in build_rl_training_instances()
        ]
        diag_objs_hv = [
            _hv_of_objs(_run_condition_on_instance(inst, frozen, 999, rl_kwargs))
            for inst in build_diagnostic_instances()
        ]
        train_hv = statistics.mean(train_objs_hv)
        diag_hv = statistics.mean(diag_objs_hv)
        gap = diag_hv - train_hv
        train_hvs.append(train_hv)
        diag_hvs.append(diag_hv)
        gaps.append(gap)
        print(f"  graine {seed} : HV_train={train_hv:.4f}  HV_diag={diag_hv:.4f}  écart={gap:+.4f}")

    print(f"\n  Moyenne sur {n_seeds} graines : "
          f"HV_train={statistics.mean(train_hvs):.4f} (+/-{statistics.pstdev(train_hvs):.4f})  "
          f"HV_diag={statistics.mean(diag_hvs):.4f} (+/-{statistics.pstdev(diag_hvs):.4f})  "
          f"écart={statistics.mean(gaps):+.4f} (+/-{statistics.pstdev(gaps):.4f})")

    print("\n=== Rapport de généralisation complet (baselines + graine de référence) ===")
    generalization_report(agents[0][1], verbose=True)

    return agents, agreements, (train_hvs, diag_hvs, gaps)


if __name__ == "__main__":
    run_campaign(n_seeds=5, n_episodes=1000, verbose=True)
