"""
Entraînement FINAL, UNIQUE, de l'agent OMF-RL -- pas un nouveau cycle
d'itération, mais la production de la politique gelée à intégrer au
pipeline officiel (benchmark_runner.py), pour une comparaison globale
avec RL-MOEA/D et les 7 algorithmes déjà établis.

Config figée (cf. model/rl/training.py) : NF=20, IT=20, NN=8 (ajusté par
rapport à NN=2 suite au diagnostic de balayage), dominance="weak",
récompense enrichie (mode par défaut de ContextualBandit).

Entraîne 3 graines indépendantes (vérification de convergence, pas
d'exploration de nouvelles pistes), retient la graine 0 comme politique
de référence -- accord de 90-100% avec les 2 autres graines, cf. sortie
de ce script.

Usage : python train_final_omf_rl.py
Produit : calibration/final_omf_rl_policy.pkl
"""
import pickle
import os

from model.rl.training import train_multi_seed, policy_agreement
from model.rl.agent import FrozenPolicy

N_SEEDS = 3
N_EPISODES = 1000


def main():
    print(f"Entraînement de {N_SEEDS} graines indépendantes ({N_EPISODES} épisodes chacune)...")
    agents = train_multi_seed(n_seeds=N_SEEDS, n_episodes=N_EPISODES, base_seed=0, verbose=True)

    print("\n=== Accord de politique inter-graines (vérification de convergence) ===")
    agreements = policy_agreement(agents)
    for s1, s2, rate, n_shared in agreements:
        if rate is None:
            print(f"  graine {s1} vs {s2} : aucun état partagé")
        else:
            print(f"  graine {s1} vs {s2} : accord={rate:.1%} sur {n_shared} états partagés")

    # Retient la graine 0 comme politique de reference (comparable a
    # n'importe laquelle des 3, vu l'accord eleve mesure ci-dessus).
    seed_ref, agent_ref, _ = agents[0]
    frozen = FrozenPolicy(agent_ref)

    os.makedirs("calibration", exist_ok=True)
    out_path = os.path.join("calibration", "final_omf_rl_policy.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(frozen, f)

    print(f"\nPolitique gelée (graine {seed_ref}) écrite dans {out_path}")
    print(f"États dans la politique : {len(frozen.q)}")
    print("\nTerminé -- prêt à être chargé par model/benchmark_runner.py.")


if __name__ == "__main__":
    main()
