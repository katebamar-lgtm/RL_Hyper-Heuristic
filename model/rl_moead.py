"""
RL-MOEA/D — MOEA/D (Zhang & Li, 2007) avec sélection adaptative d'opérateur
par bandit contextuel (UCB1), au lieu des taux de croisement/mutation fixes
de `moead.py`.

Positionnement scientifique (cf. discussion) : on ne remplace PAS les
opérateurs génétiques eux-mêmes (toujours `ga_common.shared_node_crossover`
/ `standard_mutation`, partagés avec NSGA-II/SPEA2/MOEA-D pour rester
comparable équitablement) ; seule la
DÉCISION "quel opérateur appliquer à ce sous-problème, maintenant" devient
adaptative, apprise en ligne pendant le run, plutôt que fixée a priori par
`crossover_rate`/`mutation_rate`.

Design (délibérément simple pour une première version défendable) :
  - Un agent = bandit contextuel tabulaire (PAS de réseau de neurones :
    cohérent avec la nature légère du problème, garde l'avantage de vitesse
    de MOEA/D, et se rapproche de K. Li et al. 2014 [réf. 29] plutôt que
    d'un DQN, plus dur à justifier/reproduire pour une 1ère hybridation).
  - Contexte = (bucket de stagnation du sous-problème, bucket de
    progression de la recherche). Discrétisé pour rester tabulaire.
  - Bras = 3 stratégies de reproduction, toutes construites à partir des
    opérateurs déjà partagés (`ga_common`) :
      A: croisement + mutation (comportement par défaut de moead.py)
      B: croisement seul (exploitation de la recombinaison)
      C: mutation seule (perturbation/diversification, pas de croisement)
  - Récompense = binaire (1 si le sous-problème améliore au moins un voisin
    de son voisinage B(i), 0 sinon). Choix délibéré : une récompense
    continue basée sur la magnitude du gain Tchebycheff nécessite une
    normalisation par sous-problème/instance fragile ; la récompense
    binaire est robuste, interprétable, et suffisante pour un premier
    résultat solide (peut être raffinée en V2 — cf. section "extensions").
  - Mise à jour UCB1 standard par (contexte, bras).

Extensions futures possibles, PAS incluses ici pour garder cette V1
simple à valider/ablater : récompense continue normalisée, contexte
enrichi (diversité de voisinage, delta du point idéal), bandit non
tabulaire (regression contextuelle) si le contexte devient continu.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .network import MultimodalNetwork, Connection
from .entities import Mode
from .ga_common import shared_node_crossover, standard_mutation
from .pareto import non_dominated_indices
from .budget import EvalCounter

# Réutilisation délibérée de l'infrastructure MOEA/D existante (mêmes
# structures de données, mêmes fonctions utilitaires) plutôt que
# duplication — garde RL-MOEA/D et MOEA/D strictement comparables
# opérateur pour opérateur, seule la sélection change.
from .moead import (
    Individual,
    _evaluate,
    _generate_weight_vectors,
    _neighborhoods,
    _tchebycheff,
    _initial_population,
)

N_OBJECTIVES = 4  # temps, distance, coût, correspondances

ARMS = ("crossover_mutation", "crossover_only", "mutation_only")
N_STAGNATION_BUCKETS = 3   # 0 / 1-2 / 3+ générations sans amélioration
N_PROGRESS_BUCKETS = 3     # début / milieu / fin de run


@dataclass
class RLMOEADParams:
    population_size: int = 60
    generations: int = 100
    neighborhood_size: int = 8
    max_replacements: int = 2
    max_hops: int = 40
    budget: Optional[int] = None
    seed: Optional[int] = None
    ucb_c: float = 1.4          # coefficient d'exploration UCB1
    reward_mode: str = "binary"  # "binary" (v1, defaut) | "rich" (v2, magnitude Tchebycheff -- cf. run_rl_moead)


@dataclass
class _BanditStats:
    """Compteurs UCB1 pour un contexte donné, un par bras."""
    counts: List[int] = field(default_factory=lambda: [0] * len(ARMS))
    values: List[float] = field(default_factory=lambda: [0.0] * len(ARMS))


class ContextualBandit:
    """Bandit UCB1 tabulaire, une instance de `_BanditStats` par contexte
    rencontré (créée à la demande)."""

    def __init__(self, ucb_c: float, rng: random.Random):
        self.ucb_c = ucb_c
        self.rng = rng
        self.stats: Dict[Tuple[int, int], _BanditStats] = {}

    def _get(self, context: Tuple[int, int]) -> _BanditStats:
        if context not in self.stats:
            self.stats[context] = _BanditStats()
        return self.stats[context]

    def select(self, context: Tuple[int, int]) -> int:
        s = self._get(context)
        # phase d'amorçage : chaque bras essayé au moins une fois avant UCB
        for a in range(len(ARMS)):
            if s.counts[a] == 0:
                return a
        total = sum(s.counts)
        ucb = [s.values[a] + self.ucb_c * math.sqrt(math.log(total) / s.counts[a])
               for a in range(len(ARMS))]
        best = max(ucb)
        # départage aléatoire entre bras ex-aequo (évite un biais d'ordre)
        candidates = [a for a, v in enumerate(ucb) if v == best]
        return self.rng.choice(candidates)

    def update(self, context: Tuple[int, int], arm: int, reward: float) -> None:
        s = self._get(context)
        s.counts[arm] += 1
        n = s.counts[arm]
        s.values[arm] += (reward - s.values[arm]) / n  # moyenne incrémentale

    def export_policy(self) -> Dict[str, Dict[str, float]]:
        """Sérialisable pour inspection post-run (quel bras gagne dans quel
        contexte) — utile pour l'analyse/figures de l'article."""
        out = {}
        for (stag, prog), s in self.stats.items():
            key = f"stagnation={stag},progress={prog}"
            out[key] = {ARMS[a]: round(v, 4) for a, v in enumerate(s.values)}
        return out


def _apply_arm(arm_name: str, net: MultimodalNetwork, p1: List[Connection],
                p2: List[Connection], allowed_modes: Set[Mode],
                rng: random.Random) -> Optional[List[Connection]]:
    if arm_name == "crossover_only":
        child = shared_node_crossover(p1, p2, rng)
        return child if child is not None else rng.choice([p1, p2])

    if arm_name == "mutation_only":
        base = rng.choice([p1, p2])
        mutated = standard_mutation(net, base, allowed_modes, rng)
        return mutated if mutated is not None else base

    # "crossover_mutation" (comportement par défaut MOEA/D)
    child = shared_node_crossover(p1, p2, rng)
    if child is None:
        child = rng.choice([p1, p2])
    mutated = standard_mutation(net, child, allowed_modes, rng)
    return mutated if mutated is not None else child


def _progress_bucket(counter: Optional[EvalCounter], gen: int, total_gens: int) -> int:
    """Bucket de progression -- utilise la progression du BUDGET
    (counter.count/counter.budget) quand disponible, PAS gen/total_gens.

    CORRECTION : avec generations=10**9 (convention du dépôt -- seul le
    budget doit arrêter la boucle, cf. model/calibration.py et
    model/benchmark_runner.py), gen/total_gens reste quasi toujours nul
    quel que soit le nombre réel de générations exécutées -- vérifié
    empiriquement : seul le contexte progress=0 apparaissait jamais dans
    la politique exportée, sur les trois échelles testées. L'espace de
    contexte prévu (3 stagnations x 3 progressions = 9) se réduisait donc
    en pratique à 3 (seule la stagnation variait). Repli sur gen/total_gens
    uniquement si `counter` n'a pas de budget fixé (cas non utilisé
    actuellement dans ce dépôt, mais gardé pour robustesse)."""
    if counter is not None and getattr(counter, "budget", None):
        frac = counter.count / counter.budget
    else:
        frac = gen / max(1, total_gens)
    if frac < 1 / 3:
        return 0
    if frac < 2 / 3:
        return 1
    return 2


def _stagnation_bucket(stagnation: int) -> int:
    if stagnation == 0:
        return 0
    if stagnation <= 2:
        return 1
    return 2


def run_rl_moead(net: MultimodalNetwork, origin: str, destination: str,
                  allowed_modes: Set[Mode], params: Optional[RLMOEADParams] = None,
                  counter: Optional[EvalCounter] = None,
                  return_policy: bool = False
                  ) -> List[Individual]:
    """Exécute RL-MOEA/D, retourne le front non dominé final.

    Si `return_policy=True`, retourne un tuple (front, policy_dict) —
    pratique pour inspecter/figurer la politique apprise sans relancer un
    run séparé.
    """
    params = params or RLMOEADParams()
    rng = random.Random(params.seed)
    counter = counter if counter is not None else EvalCounter(budget=params.budget)

    pop = _initial_population(net, origin, destination, allowed_modes,
                               params.population_size, params.max_hops, rng, counter)
    if not pop:
        return ([], {}) if return_policy else []
    n = len(pop)

    weights = _generate_weight_vectors(n, N_OBJECTIVES, rng)
    neighborhoods = _neighborhoods(weights, min(params.neighborhood_size, n))
    ideal = [min(ind.objectives[k] for ind in pop) for k in range(N_OBJECTIVES)]

    bandit = ContextualBandit(params.ucb_c, rng)
    stagnation = [0] * n  # générations consécutives sans amélioration, par sous-problème

    for gen in range(params.generations):
        if counter.exhausted():
            break
        progress = _progress_bucket(counter, gen, params.generations)

        for i in range(n):
            if counter.exhausted():
                break
            context = (_stagnation_bucket(stagnation[i]), progress)
            arm = bandit.select(context)
            arm_name = ARMS[arm]

            b_i = neighborhoods[i]
            j1, j2 = rng.sample(b_i, 2) if len(b_i) >= 2 else (i, i)
            child = _apply_arm(arm_name, net, pop[j1].path, pop[j2].path, allowed_modes, rng)

            child_obj = _evaluate(net, child, counter)
            ideal = [min(ideal[k], child_obj[k]) for k in range(N_OBJECTIVES)]
            # g_before et g_child_i évalués avec le MEME point idéal (mis à
            # jour), pour une comparaison cohérente -- nécessaire pour le
            # mode "rich" ci-dessous (mesure la magnitude de l'amélioration
            # du point de vue du sous-problème i lui-même, celui dont la
            # décision d'opérateur est récompensée).
            g_before = _tchebycheff(pop[i].objectives, weights[i], ideal)
            g_child_i = _tchebycheff(child_obj, weights[i], ideal)

            replacements = 0
            order = list(b_i)
            rng.shuffle(order)
            for j in order:
                if replacements >= params.max_replacements:
                    break
                g_child = _tchebycheff(child_obj, weights[j], ideal)
                g_current = _tchebycheff(pop[j].objectives, weights[j], ideal)
                if g_child <= g_current:
                    pop[j] = Individual(child, child_obj)
                    replacements += 1

            if params.reward_mode == "rich":
                # Magnitude de l'amelioration Tchebycheff relative, du point
                # de vue du sous-probleme i -- signal continu, TOUJOURS
                # disponible (jamais clairseme comme le binaire), directement
                # aligne avec ce que l'algorithme optimise reellement (la
                # scalarisation), plutot qu'un evenement de remplacement
                # binaire qui peut favoriser des ameliorations "faciles"
                # (ex. crossover_only) sans rapport avec la qualite globale
                # du front (cf. diagnostic post-campagne).
                reward = max(0.0, (g_before - g_child_i) / (g_before + 1e-9))
            else:  # "binary" (v1, comportement d'origine, conserve pour ablation)
                reward = 1.0 if replacements > 0 else 0.0

            bandit.update(context, arm, reward)
            stagnation[i] = 0 if replacements > 0 else stagnation[i] + 1

    objs = [ind.objectives for ind in pop]
    nd_idx = non_dominated_indices(objs)
    front = [pop[i] for i in nd_idx]
    return (front, bandit.export_policy()) if return_policy else front


# ---------------------------------------------------------------------------
# Variante d'ablation : même boucle, même 3 bras, mais sélection UNIFORME
# ALÉATOIRE au lieu du bandit UCB1. Isole l'apport du RL (apprentissage en
# ligne) de celui de la simple diversité d'opérateurs à chaque génération.
# Un reviewer demandera systématiquement cette comparaison — cf. discussion.
#
# NOTE (nettoyage) : ce module contenait auparavant DEUX définitions de
# `run_rl_moead_random` (une utilisant RLMOEADParams, une seconde -- celle
# conservée ici -- utilisant RLMOEADRandomParams dédié). La seconde
# écrasait silencieusement la première en Python (code mort, jamais
# exécuté) ; supprimée pour éviter toute confusion, aucun changement de
# comportement (la version active était déjà celle-ci).
# ---------------------------------------------------------------------------

@dataclass
class RLMOEADRandomParams:
    population_size: int = 60
    generations: int = 100
    neighborhood_size: int = 8
    max_replacements: int = 2
    max_hops: int = 40
    budget: Optional[int] = None
    seed: Optional[int] = None


def run_rl_moead_random(net: MultimodalNetwork, origin: str, destination: str,
                         allowed_modes: Set[Mode],
                         params: Optional[RLMOEADRandomParams] = None,
                         counter: Optional[EvalCounter] = None
                         ) -> List[Individual]:
    """Baseline d'ablation pour RL-MOEA/D : bras tiré uniformément au hasard
    à chaque sous-problème/génération, sans aucun apprentissage. Mêmes
    opérateurs, même budget, même protocole — seule la logique de décision
    change. Si RL-MOEA/D ne bat pas significativement cette variante, le
    gain observé n'est pas attribuable au RL."""
    params = params or RLMOEADRandomParams()
    rng = random.Random(params.seed)
    counter = counter if counter is not None else EvalCounter(budget=params.budget)

    pop = _initial_population(net, origin, destination, allowed_modes,
                               params.population_size, params.max_hops, rng, counter)
    if not pop:
        return []
    n = len(pop)

    weights = _generate_weight_vectors(n, N_OBJECTIVES, rng)
    neighborhoods = _neighborhoods(weights, min(params.neighborhood_size, n))
    ideal = [min(ind.objectives[k] for ind in pop) for k in range(N_OBJECTIVES)]

    for _gen in range(params.generations):
        if counter.exhausted():
            break
        for i in range(n):
            if counter.exhausted():
                break
            arm_name = rng.choice(ARMS)
            b_i = neighborhoods[i]
            j1, j2 = rng.sample(b_i, 2) if len(b_i) >= 2 else (i, i)
            child = _apply_arm(arm_name, net, pop[j1].path, pop[j2].path, allowed_modes, rng)

            child_obj = _evaluate(net, child, counter)
            ideal = [min(ideal[k], child_obj[k]) for k in range(N_OBJECTIVES)]

            replacements = 0
            order = list(b_i)
            rng.shuffle(order)
            for j in order:
                if replacements >= params.max_replacements:
                    break
                g_child = _tchebycheff(child_obj, weights[j], ideal)
                g_current = _tchebycheff(pop[j].objectives, weights[j], ideal)
                if g_child <= g_current:
                    pop[j] = Individual(child, child_obj)
                    replacements += 1

    objs = [ind.objectives for ind in pop]
    nd_idx = non_dominated_indices(objs)
    return [pop[i] for i in nd_idx]
