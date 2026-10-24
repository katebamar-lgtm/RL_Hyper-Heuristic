"""
Construction de l'état observé par l'agent RL avant chaque décision
largeur/profondeur, à l'échelle d'UN voisin (cf. model/omf_standalone.py,
tirage R par voisin, conforme au flowchart).

État à 5 caractéristiques discrétisées (version réduite du protocole —
volontairement simple pour la première itération d'implémentation ; les
features 7-8 du protocole complet, dédiées au transfert d'échelle, sont
ici fusionnées dans `scale_bucket_from_nodes`). Chaque composante est
bucketisée en un nombre restreint d'intervalles pour rester exploitable
par un Q-learning tabulaire / bandit contextuel à table de valeurs.
"""

from __future__ import annotations
from typing import Optional, Tuple

from ..network import MultimodalNetwork
from ..budget import EvalCounter


def _bucket(x: float, n_buckets: int) -> int:
    """Discrétise x (attendu dans [0,1], tolère un léger dépassement) en
    un entier dans [0, n_buckets-1]."""
    x = max(0.0, min(1.0, x))
    idx = int(x * n_buckets)
    return min(idx, n_buckets - 1)


State = Tuple[int, int, int]


def scale_bucket_from_nodes(n_nodes: int) -> int:
    """Bucket d'échelle du réseau, fondé sur des SEUILS DE TAILLE explicites
    et alignés sur les conventions déjà en usage dans le dépôt : small
    <=~25-30 noeuds (calibration/final tests/entraînement RL), medium
    ~40-65, large au-delà -- cf. model/benchmark_instances.py,
    model/rl/train_instances.py, model/rl/diagnostic_instances.py.

    Remplace une première version fondée sur un ratio log(n)/log(n_ref)
    qui SATURAIT les buckets aux tailles réellement utilisées (vérifié
    empiriquement : les 6 instances d'entraînement, de tailles 18 à 110,
    ne couvraient que 2 des 3 buckets, les instances small et medium étant
    toutes deux écrasées dans le même bucket -- limitait artificiellement
    l'espace d'état appris).

    Les réseaux beaucoup plus grands qu'à l'entraînement (ex. Lyon, 696
    noeuds) tombent dans le même bucket "large" que les plus grandes
    instances d'entraînement -- c'est le régime d'EXTRAPOLATION
    précisément visé par la question de recherche RQ1 du protocole
    expérimental (la politique apprise sur small/medium/large
    synthétiques généralise-t-elle au-delà de cette gamme ?), pas une
    limite à corriger : avec seulement 3 buckets, on ne peut de toute
    façon pas distinguer "large" de "beaucoup plus grand que large" sans
    perdre la comparabilité avec les buckets appris à l'entraînement.
    """
    if n_nodes <= 30:
        return 0
    elif n_nodes <= 70:
        return 1
    else:
        return 2


def build_state(net: MultimodalNetwork,
                 fi_obj,
                 counter: Optional[EvalCounter],
                 it_idx: int,
                 it_total: int,
                 recent_success_width: float,
                 recent_success_depth: float,
                 scale_bkt: int,
                 n_buckets: int = 3) -> State:
    """État = (progression au sein du redémarrage courant, succès récent
    largeur/profondeur combiné en un seul indicateur relatif, échelle du
    réseau).

    `it_idx`/`it_total` (progression LOCALE, au sein du redémarrage NF
    courant) remplace une première version fondée sur la progression du
    budget GLOBAL (`counter.count / counter.budget`), qui restait quasi
    constante sur toute la durée d'un run avec les hyperparamètres
    calibrés (NF x IT x NN << budget total disponible) -- vérifié
    empiriquement : cette feature ne variait jamais au-delà de son bucket
    minimal, ce qui limitait artificiellement l'espace d'état appris.
    `it_idx/it_total` balaie tout l'intervalle [0,1] à CHAQUE exécution,
    quel que soit le budget global.

    Version volontairement compacte (3 composantes) pour limiter la
    taille de la table Q dès la première implémentation ; à enrichir
    seulement si les ablations du protocole (stagnation, diversité de
    front...) montrent un gain net -- cf. §7 du protocole expérimental.
    """
    local_progress = 0.0 if it_total <= 1 else it_idx / (it_total - 1)

    # indicateur relatif largeur vs profondeur, dans [0,1] : 0.5 = aucune
    # préférence apprise pour l'instant (valeur initiale symétrique).
    denom = recent_success_width + recent_success_depth
    relative_success = 0.5 if denom == 0 else recent_success_width / denom

    return (
        _bucket(local_progress, n_buckets),
        _bucket(relative_success, n_buckets),
        scale_bkt,
    )
