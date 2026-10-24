"""
Boucle d'entraînement de l'agent RL (ContextualBandit) sur le jeu
d'instances dédié (model/rl/train_instances.py), via le pilote OMF
autonome (model/omf_standalone.py).

Même règle de budget que le reste du dépôt : budget = 300 x population_size
(cf. model/benchmark_instances.py, commentaire `_BUDGET_MULTIPLIER`),
appliquée ici à chaque épisode d'entraînement pour rester cohérente avec
le budget utilisé en évaluation finale.
"""

from __future__ import annotations
import random
from typing import List, Optional, Tuple

from ..budget import EvalCounter
from ..omf_standalone import run_omf
from .agent import ContextualBandit
from .train_instances import build_rl_training_instances, max_training_n_nodes, ALL_MODES

_BUDGET_MULTIPLIER = 300
_POP_SIZE_BY_CATEGORY = {"small": 30, "medium": 50, "large": 80}

# (nf, nn, it) calibrés (cf. calibration/final_omf_standalone_multiscale.json,
# obtenu via model/rl/calibrate_omf_standalone.py sur
# build_multiscale_tuning_instances()) -- SAUF nn, fixé à 8 (au lieu de 2)
# suite au diagnostic de balayage NN : NN=2 (calibré pour la version
# STATIQUE) limitait fortement le levier de décision de l'agent ; NN=8 a
# montré un gain net de généralisation (0.34 -> 0.55 sur le diagnostic de
# développement) sans recalibration complète -- retenu comme réglage
# final pragmatique (une seule campagne, pas de nouveau cycle d'itération,
# cf. discussion). `dominance="weak"` retenu à la calibration multi-échelle.
DEFAULT_NF = 20
DEFAULT_NN = 8
DEFAULT_IT = 20
DEFAULT_DOMINANCE = "weak"


def train(n_episodes: int = 300,
          nf: int = DEFAULT_NF, nn: int = DEFAULT_NN, it: int = DEFAULT_IT,
          dominance: str = DEFAULT_DOMINANCE,
          epsilon_start: float = 0.3, epsilon_end: float = 0.02,
          epsilon_decay_steps: Optional[int] = None,
          alpha: Optional[float] = None,
          seed: int = 0, verbose: bool = True) -> Tuple[ContextualBandit, List[dict]]:
    """Entraîne un ContextualBandit en alternant aléatoirement entre les
    instances d'entraînement. Retourne l'agent entraîné et un journal
    (liste de dicts) pour inspection/tracé ultérieur (HV par épisode,
    instance utilisée, etc.).

    `epsilon_decay_steps=None` (par défaut) -> calé automatiquement sur la
    moitié du nombre de DÉCISIONS attendu pour `n_episodes` (estimé à
    nf*it*nn décisions/épisode) -- epsilon atteint epsilon_end à mi-course
    de l'entraînement, puis y reste. `alpha=None` -> 1/N(s,a) (cf.
    ContextualBandit) ; `alpha=<float>` reproduit l'ancien comportement
    (moyenne mobile fixe), pour ablation uniquement.
    """
    instances = build_rl_training_instances()
    n_nodes_ref = max_training_n_nodes()
    rng_master = random.Random(seed)

    if epsilon_decay_steps is None:
        epsilon_decay_steps = max(1, (n_episodes * nf * it * nn) // 2)

    agent = ContextualBandit(n_nodes_ref=n_nodes_ref,
                              epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                              epsilon_decay_steps=epsilon_decay_steps, alpha=alpha,
                              rng=random.Random(seed))
    # Mémorise la configuration d'entraînement sur l'agent lui-même --
    # CORRECTION d'un bug trouvé lors d'une ablation manuelle (NN=8 à
    # l'entraînement) : `generalization_report`/`run_campaign` évaluaient
    # jusqu'ici l'agent avec les valeurs DEFAULT_* (donc NN=2) quel que
    # soit le NN réellement utilisé à l'entraînement, faussant silencieusement
    # toute évaluation faite avec une configuration non standard.
    agent.trained_config = dict(nf=nf, nn=nn, it=it, dominance=dominance)

    log: List[dict] = []
    # les réseaux sont coûteux à générer -> on les construit une fois et on
    # les réutilise à travers les épisodes (déterministe, cf. generate_network).
    built = [(inst, inst.build()) for inst in instances]

    for episode in range(n_episodes):
        inst, net = rng_master.choice(built)
        origin, destination = inst.endpoints(net)
        pop_size = _POP_SIZE_BY_CATEGORY[inst.category]
        budget = _BUDGET_MULTIPLIER * pop_size
        counter = EvalCounter(budget=budget)

        episode_rng = random.Random(rng_master.randint(0, 2**31 - 1))
        paths, objs = run_omf(net, origin, destination, ALL_MODES,
                               nf=nf, nn=nn, it=it, dominance=dominance,
                               counter=counter, controller=agent, rng=episode_rng)

        log.append({
            "episode": episode,
            "instance": inst.name,
            "category": inst.category,
            "n_front": len(paths),
            "evals_used": counter.count,
        })

        if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
            print(f"[train] épisode {episode + 1}/{n_episodes} "
                  f"instance={inst.name} front={len(paths)} evals={counter.count}")

    return agent, log


def train_multi_seed(n_seeds: int = 5, n_episodes: int = 300,
                      nf: int = DEFAULT_NF, nn: int = DEFAULT_NN, it: int = DEFAULT_IT,
                      dominance: str = DEFAULT_DOMINANCE,
                      epsilon_start: float = 0.3, epsilon_end: float = 0.02,
                      epsilon_decay_steps: Optional[int] = None,
                      alpha: Optional[float] = None,
                      base_seed: int = 0, verbose: bool = True
                      ) -> List[Tuple[int, ContextualBandit, List[dict]]]:
    """Entraîne `n_seeds` agents INDÉPENDANTS (graines base_seed, base_seed+1,
    ...), chacun sur l'intégralité du jeu d'instances d'entraînement.
    Nécessaire pour estimer la variance liée à l'apprentissage lui-même
    (cf. protocole expérimental §7, point 5 -- absente des algorithmes sans composante apprise
    puisqu'elle ne comportait pas de composante apprise).

    Retourne [(seed, agent, log), ...]."""
    results = []
    for i in range(n_seeds):
        seed = base_seed + i
        if verbose:
            print(f"\n=== Entraînement graine {i + 1}/{n_seeds} (seed={seed}) ===")
        agent, log = train(n_episodes=n_episodes, nf=nf, nn=nn, it=it, dominance=dominance,
                            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                            epsilon_decay_steps=epsilon_decay_steps, alpha=alpha,
                            seed=seed, verbose=verbose)
        results.append((seed, agent, log))
    return results


def policy_agreement(agents: List[Tuple[int, ContextualBandit, List[dict]]]
                      ) -> List[Tuple[int, int, Optional[float], int]]:
    """Taux d'accord PAIRWISE entre les politiques gloutonnes de plusieurs
    agents entraînés avec des graines différentes, restreint aux états
    visités par LES DEUX agents de chaque paire (un état visité par un
    seul des deux n'est comparable à rien).

    Retourne [(seed_a, seed_b, taux_accord_ou_None, nb_etats_partages), ...].
    `taux_accord=None` si aucun état n'est partagé entre les deux agents
    (ex. très peu d'épisodes d'entraînement) -- à distinguer explicitement
    d'un désaccord total (taux_accord=0.0), pour ne pas conclure à tort
    à une politique instable alors qu'elle n'a simplement pas encore assez
    d'états communs pour être comparée.
    """
    from itertools import combinations
    results = []
    for (s1, a1, _), (s2, a2, _) in combinations(agents, 2):
        p1, p2 = a1.policy_snapshot(), a2.policy_snapshot()
        shared = set(p1) & set(p2)
        if not shared:
            results.append((s1, s2, None, 0))
            continue
        agree = sum(1 for st in shared if p1[st] == p2[st])
        results.append((s1, s2, agree / len(shared), len(shared)))
    return results


if __name__ == "__main__":
    # Lancement rapide en ligne de commande pour une première vérification
    # locale (VS Code) : `python -m model.rl.training` depuis la racine du
    # dépôt. Réduit le nombre d'épisodes pour un test de fumée rapide.
    trained_agent, training_log = train(n_episodes=60, verbose=True)
    n_states = len(trained_agent.q)
    print(f"\nEntraînement terminé : {n_states} états distincts visités.")
    print("Exemples de politique apprise (état -> action) :")
    for state, action in list(trained_agent.policy_snapshot().items())[:10]:
        print(f"  {state} -> {action}")
