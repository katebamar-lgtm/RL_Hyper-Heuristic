"""
Agents de contrôle largeur/profondeur, partagés entre deux métaheuristiques
hôtes :

  - OMF (model/omf_standalone.py)  : contrôle du voisinage largeur/profondeur
    par voisin généré.
  - MOEA/D (model/moead_rl.py)     : contrôle de l'opérateur de mutation
    (largeur/profondeur, réutilisés d'OMF) par sous-problème/génération.

Toutes les classes de ce module sont conformes à l'interface attendue par
la boucle appelante correspondante :

    build_state(...) -> état hashable   [signature propre à chaque hôte]
    choose(state) -> "width" | "depth"
    observe(state, action, reward, next_state) -> None

Le cœur du bandit tabulaire (Q(s,a), epsilon décroissant, alpha=1/N(s,a))
est FACTORISÉ dans `_TabularBanditCore`, partagé sans duplication entre
`ContextualBandit` (OMF) et `MOEADContextualBandit` (MOEA/D) -- seule la
construction de l'état (`build_state`) diffère, chacune adaptée aux
informations disponibles dans sa boucle hôte respective.

REFACTORISATION (introduction de MOEA/D+RL) : le comportement de
`ContextualBandit`/`RandomController`/`FrozenPolicy` pour OMF est
VÉRIFIÉ IDENTIQUE bit-à-bit à la version précédente (même séquence
d'appels rng, mêmes calculs) -- cf. test de non-régression avant livraison.
"""

from __future__ import annotations
import random
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from ..network import MultimodalNetwork
from ..budget import EvalCounter
from .state import State, build_state, scale_bucket_from_nodes

ACTIONS = ("width", "depth")


# ==================================================================== #
# Cœur partagé : Q-learning/bandit tabulaire, indépendant de la
# métaheuristique hôte (ne connaît ni OMF ni MOEA/D).
# ==================================================================== #

class _TabularBanditCore:
    """Q(s,a) mis à jour par Q(s,a) <- Q(s,a) + alpha*(reward - Q(s,a)),
    alpha=1/N(s,a) par défaut (moyenne empirique exacte, convergente --
    cf. correction historique : une moyenne mobile à taux fixe ne converge
    JAMAIS, quelle que soit la quantité de données, et a été la cause
    principale d'un accord de politique inter-graines de seulement 57%
    avant correction). Exploration epsilon-greedy à décroissance linéaire.

    Ne définit PAS `build_state` (propre à chaque métaheuristique hôte,
    cf. `ContextualBandit`/`MOEADContextualBandit`) ni de récompense
    spécifique -- seule la mécanique générique de bandit est ici.
    """

    def __init__(self,
                 epsilon_start: float = 0.3, epsilon_end: float = 0.02,
                 epsilon_decay_steps: int = 400_000,
                 alpha: Optional[float] = None,
                 k_history: int = 20, rng: Optional[random.Random] = None):
        """`alpha=None` (par défaut) -> 1/N(s,a). `alpha=<float>` ->
        moyenne mobile fixe (ancien comportement, ablation uniquement)."""
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.fixed_alpha = alpha
        self.decision_count = 0
        self.rng = rng or random.Random()
        self.q: Dict[State, Dict[str, float]] = defaultdict(lambda: {"width": 0.0, "depth": 0.0})
        self.visit_counts: Dict[State, Dict[str, int]] = defaultdict(lambda: {"width": 0, "depth": 0})
        self.hist_width: Deque[float] = deque(maxlen=k_history)
        self.hist_depth: Deque[float] = deque(maxlen=k_history)
        self.action_log: list = []   # [(bucket_progression, action), ...]

    @property
    def epsilon(self) -> float:
        """Décroissance linéaire de epsilon_start à epsilon_end sur
        epsilon_decay_steps DÉCISIONS (pas épisodes/générations)."""
        frac = min(1.0, self.decision_count / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _relative_success(self) -> Tuple[float, float]:
        succ_w = sum(self.hist_width) / len(self.hist_width) if self.hist_width else 0.5
        succ_d = sum(self.hist_depth) / len(self.hist_depth) if self.hist_depth else 0.5
        return succ_w, succ_d

    def choose(self, state: State) -> str:
        self.decision_count += 1
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(ACTIONS)
        else:
            qs = self.q[state]
            action = "width" if qs["width"] >= qs["depth"] else "depth"
        self.action_log.append((state[0], action))   # state[0] = bucket de progression
        return action

    def observe(self, state: State, action: str, reward: float, next_state: State) -> None:
        self.visit_counts[state][action] += 1
        alpha = self.fixed_alpha if self.fixed_alpha is not None else 1.0 / self.visit_counts[state][action]
        qs = self.q[state]
        qs[action] += alpha * (reward - qs[action])
        (self.hist_width if action == "width" else self.hist_depth).append(reward)

    def reset_episode_history(self) -> None:
        """À appeler entre deux épisodes d'ENTRAÎNEMENT si l'on souhaite
        que le taux de succès glissant ne persiste pas d'une instance
        d'entraînement à l'autre (optionnel, cf. ablation possible)."""
        self.hist_width.clear()
        self.hist_depth.clear()

    def policy_snapshot(self) -> Dict[State, str]:
        """Politique gloutonne actuelle, utile pour l'analyse qualitative
        (heatmap état -> action) demandée dans le protocole expérimental."""
        return {s: ("width" if v["width"] >= v["depth"] else "depth")
                for s, v in self.q.items()}

    def action_rate_by_progress(self) -> Dict[int, Dict[str, float]]:
        """Taux d'action {'width':.., 'depth':..} par bucket de progression
        -- détecte une politique dégénérée et visualise la structure
        apprise (cf. protocole expérimental §6)."""
        from collections import Counter
        by_bucket: Dict[int, Counter] = {}
        for bucket, action in self.action_log:
            by_bucket.setdefault(bucket, Counter())[action] += 1
        return {b: {a: c / sum(cnt.values()) for a, c in cnt.items()}
                for b, cnt in by_bucket.items()}


# ==================================================================== #
# Contrôle négatif -- généralisé pour accepter la signature de
# build_state de N'IMPORTE QUELLE métaheuristique hôte (OMF ou MOEA/D) :
# le choix est uniforme et ne dépend d'aucun argument de toute façon.
# ==================================================================== #

class RandomController:
    """Contrôle négatif (§3 du protocole expérimental) : choix uniforme,
    aucun apprentissage. `build_state` accepte des arguments arbitraires
    pour rester utilisable tel quel avec OMF ET MOEA/D sans sous-classe
    dédiée -- l'état renvoyé est de toute façon un état factice, jamais
    utilisé pour décider."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def build_state(self, *args, **kwargs) -> State:
        return (0, 0, 0)   # état factice, jamais utilisé pour décider

    def choose(self, state: State) -> str:
        return self.rng.choice(ACTIONS)

    def observe(self, state: State, action: str, reward: float, next_state: State) -> None:
        pass  # pas d'apprentissage, par construction


# ==================================================================== #
# Contrôleur OMF (model/omf_standalone.py)
# ==================================================================== #

class ContextualBandit(_TabularBanditCore):
    """Bandit contextuel pour OMF : état = (progression LOCALE au sein du
    redémarrage courant, succès récent largeur/profondeur, échelle du
    réseau) -- cf. model/rl/state.py::build_state.

    `n_nodes_ref` DOIT être fixé une fois pour toutes à partir du jeu
    d'entraînement, puis ne plus jamais être modifié -- y compris pendant
    l'évaluation sur les instances de test/Lyon (cf. protocole
    expérimental §3).
    """

    def __init__(self, n_nodes_ref: int,
                 epsilon_start: float = 0.3, epsilon_end: float = 0.02,
                 epsilon_decay_steps: int = 400_000,
                 alpha: Optional[float] = None,
                 reward_mode: str = "rich",
                 k_history: int = 20, rng: Optional[random.Random] = None):
        super().__init__(epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                          epsilon_decay_steps=epsilon_decay_steps, alpha=alpha,
                          k_history=k_history, rng=rng)
        self.n_nodes_ref = n_nodes_ref
        self.reward_mode = reward_mode

    def compute_reward(self, v_dominates_fi: bool, v_obj, fi_obj) -> float:
        """`reward_mode="rich"` (défaut) : 1.0 si dominance effective,
        sinon crédit partiel (jusqu'à 0.5) proportionnel à la fraction
        d'objectifs non dégradés -- remplace une récompense 0/1 pure,
        statistiquement rare en fin de recherche locale.
        `reward_mode="binary"` : 1.0/0.0 pur (ablation)."""
        if v_dominates_fi:
            return 1.0
        if self.reward_mode == "binary":
            return 0.0
        m = len(fi_obj)
        not_worse = sum(1 for k in range(m) if v_obj[k] <= fi_obj[k])
        return 0.5 * (not_worse / m)

    def build_state(self, net: MultimodalNetwork, fi_obj, counter: Optional[EvalCounter],
                     it_idx: int = 0, it_total: int = 1) -> State:
        succ_w, succ_d = self._relative_success()
        scale_bkt = scale_bucket_from_nodes(len(net.nodes))
        return build_state(net, fi_obj, counter, it_idx, it_total, succ_w, succ_d, scale_bkt)


class FrozenPolicy:
    """Politique GELÉE d'un ContextualBandit (OMF) entraîné : exploite
    uniquement, n'apprend plus. À utiliser pour TOUTE évaluation/comparaison
    -- distingue le budget d'APPRENTISSAGE (consommé une fois par
    `train()`) du budget d'INFÉRENCE (protocole expérimental §5).

    Repli documenté pour un état jamais rencontré à l'entraînement :
    action "width" (choix empiriquement dominant dans ce domaine)."""

    def __init__(self, trained_agent: "ContextualBandit"):
        # dict SIMPLE (pas defaultdict) : FrozenPolicy ne fait que lire
        # (.get()), jamais écrire -- nécessaire pour que l'objet reste
        # sérialisable (pickle), une defaultdict avec lambda locale ne
        # l'est pas (cf. train_final_omf_rl.py, bug trouvé à la livraison).
        self.q = dict(trained_agent.q)
        self.n_nodes_ref = trained_agent.n_nodes_ref
        succ_w, succ_d = trained_agent._relative_success()
        self._succ_w, self._succ_d = succ_w, succ_d
        self.n_unseen_states = 0
        self.trained_config = getattr(trained_agent, "trained_config", None)

    def build_state(self, net: MultimodalNetwork, fi_obj, counter: Optional[EvalCounter],
                     it_idx: int = 0, it_total: int = 1) -> State:
        scale_bkt = scale_bucket_from_nodes(len(net.nodes))
        return build_state(net, fi_obj, counter, it_idx, it_total, self._succ_w, self._succ_d, scale_bkt)

    def choose(self, state: State) -> str:
        qs = self.q.get(state)
        if qs is None:
            self.n_unseen_states += 1
            return "width"
        return "width" if qs["width"] >= qs["depth"] else "depth"

    def observe(self, state: State, action: str, reward: float, next_state: State) -> None:
        pass  # gelé : aucun apprentissage pendant l'évaluation


# ==================================================================== #
# Contrôleur MOEA/D (model/moead_rl.py) -- NOUVEAU
# ==================================================================== #

class MOEADContextualBandit(_TabularBanditCore):
    """Bandit contextuel pour MOEA/D : état = (progression en générations,
    succès récent largeur/profondeur, échelle du réseau, RÉGION DU
    SOUS-PROBLÈME [objectif dominant du vecteur de poids]) -- cf.
    model/rl/state_moead.py::build_state_moead.

    Contrairement à OMF, la récompense n'est PAS calculée par une méthode
    dédiée sur ce contrôleur : `model/moead_rl.py` la calcule directement
    (amélioration relative de la scalarisation de Tchebycheff, signal
    continu déjà disponible dans la boucle MOEA/D -- pas de proxy à
    inventer, contrairement à OMF) et appelle `observe()` avec cette
    valeur telle quelle.
    """

    def __init__(self, n_nodes_ref: int,
                 epsilon_start: float = 0.3, epsilon_end: float = 0.02,
                 epsilon_decay_steps: int = 400_000,
                 alpha: Optional[float] = None,
                 k_history: int = 20, rng: Optional[random.Random] = None):
        super().__init__(epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                          epsilon_decay_steps=epsilon_decay_steps, alpha=alpha,
                          k_history=k_history, rng=rng)
        self.n_nodes_ref = n_nodes_ref

    def build_state(self, net: MultimodalNetwork, weight, gen_idx: int, gen_total: int,
                     counter: Optional[EvalCounter] = None) -> State:
        from .state_moead import build_state_moead
        succ_w, succ_d = self._relative_success()
        scale_bkt = scale_bucket_from_nodes(len(net.nodes))
        return build_state_moead(counter, gen_idx, gen_total, succ_w, succ_d, scale_bkt, weight)


class MOEADFrozenPolicy:
    """Politique GELÉE d'un MOEADContextualBandit entraîné -- même
    principe que `FrozenPolicy` (OMF), adaptée à la signature de
    `build_state` de MOEA/D (vecteur de poids + génération, pas de
    chemin/objectif courant `fi_obj`)."""

    def __init__(self, trained_agent: "MOEADContextualBandit"):
        self.q = trained_agent.q
        self.n_nodes_ref = trained_agent.n_nodes_ref
        succ_w, succ_d = trained_agent._relative_success()
        self._succ_w, self._succ_d = succ_w, succ_d
        self.n_unseen_states = 0
        self.trained_config = getattr(trained_agent, "trained_config", None)

    def build_state(self, net: MultimodalNetwork, weight, gen_idx: int, gen_total: int,
                     counter: Optional[EvalCounter] = None) -> State:
        from .state_moead import build_state_moead
        scale_bkt = scale_bucket_from_nodes(len(net.nodes))
        return build_state_moead(counter, gen_idx, gen_total, self._succ_w, self._succ_d, scale_bkt, weight)

    def choose(self, state: State) -> str:
        qs = self.q.get(state)
        if qs is None:
            self.n_unseen_states += 1
            return "width"
        return "width" if qs["width"] >= qs["depth"] else "depth"

    def observe(self, state: State, action: str, reward: float, next_state: State) -> None:
        pass
