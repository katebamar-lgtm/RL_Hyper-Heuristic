"""
OMF autonome (« OMF seul »), fidèle au flowchart fourni par l'auteur du
projet — distinct de `model/omf.py` qui reste inchangé (utilisé tel quel
comme opérateur de mutation par un algorithme génétique hybride antérieur, non repris ici).

Différences avec `model/omf.py::omf()`, confirmées contre le flowchart :

  1. Boucle externe "For i = 0 to NF" : NF démarrages indépendants, chacun
     initialisé par un chemin aléatoire Fi, TOUS accumulés dans une
     archive NonDom commune. `model/omf.py::omf()` ne fait qu'un seul
     démarrage et retourne un chemin unique, pas un front.
  2. Tirage de R : le flowchart place le losange "R < rand ?" À L'INTÉRIEUR
     de la boucle "For j = 0 to NN" -> un tirage indépendant PAR VOISIN
     généré, alors que `omf.py::omf()` tire une seule fois par itération
     IT et applique ce choix aux NN voisins de cette itération.

Ce module réutilise uniquement les fonctions utilitaires pures déjà
présentes et validées dans `model/omf.py` et `model/pareto.py`
(neighborhood_width, neighborhood_depth, random_walk_path, dominates,
strictly_dominates, non_dominated_indices) : aucune duplication de cette
logique, seule la boucle de contrôle change.
"""

from __future__ import annotations
import random
from typing import List, Optional, Sequence, Set, Tuple

from .network import MultimodalNetwork, Connection
from .entities import Mode
from .omf import random_walk_path, neighborhood_width, neighborhood_depth
from .pareto import dominates, strictly_dominates, non_dominated_indices
from .budget import EvalCounter

# Interface attendue d'un "controller" RL (cf. model/rl/agent.py) :
#   controller.build_state(net, fi_obj, counter, it_idx, it_total) -> état (hashable)
#   controller.choose(state) -> "width" | "depth"
#   controller.observe(state, action, reward, next_state) -> None
#   controller.compute_reward(v_dominates_fi, v_obj, fi_obj) -> float  [OPTIONNEL]
# `controller=None` reproduit le comportement par défaut du flowchart
# (tirage R < rand par voisin).
# `it_idx`/`it_total` : progression au sein du redémarrage COURANT (0-based,
# de 0 à it_total-1), transmis en plus du budget global -- cf.
# model/rl/state.py, remplace un premier indicateur (progression du
# budget global) qui restait quasi constant sur toute la durée d'un run
# avec les hyperparamètres calibrés (NF/IT/NN << budget total).
# `compute_reward` (optionnel, absent -> repli sur la récompense binaire
# 1.0/0.0 de dominance) : permet à un contrôleur de définir une
# récompense plus riche que la seule dominance -- cf. model/rl/agent.py,
# `ContextualBandit.compute_reward` (récompense graduée, cf. protocole
# expérimental §2.4, ablation "récompense riche vs pauvre").


def _eval(net: MultimodalNetwork, path: List[Connection],
          counter: Optional[EvalCounter]) -> Tuple[float, ...]:
    if counter is not None:
        return counter.evaluate(net, path)
    return net.evaluate_route(path).as_tuple()


def run_omf(net: MultimodalNetwork, sn: str, dn: str,
            allowed_modes: Set[Mode],
            nf: int = 5, nn: int = 8, it: int = 15, r: float = 0.5,
            dominance: str = "weak",
            rng: Optional[random.Random] = None,
            counter: Optional[EvalCounter] = None,
            controller=None,
            ) -> Tuple[List[List[Connection]], List[Tuple[float, ...]]]:
    """Algorithme du flowchart fourni : NF démarrages, IT itérations, NN
    voisins par itération, choix largeur/profondeur tiré PAR VOISIN.

    Retourne (chemins, objectifs) filtrés à la non-dominance GLOBALE sur
    l'ensemble des NF démarrages (le flowchart ne le montre pas
    explicitement au niveau du schéma, mais c'est nécessaire : sans ce
    filtrage final, NonDom peut contenir des solutions d'un démarrage i1
    dominées par une solution trouvée au démarrage i2, ce qui fausserait
    tout calcul ultérieur de HV/IGD).

    `controller`, si fourni, remplace le tirage `r < rng.random()` par une
    décision apprise à chaque génération de voisin (cf. model/rl/agent.py).
    Sans lui, le comportement est celui du flowchart original.
    """
    rng = rng or random.Random()
    dom_fn = strictly_dominates if dominance == "strict" else dominates

    non_dom: List[Tuple[List[Connection], Tuple[float, ...]]] = []

    for _ in range(nf):                                    # For i = 0 to NF
        if counter is not None and counter.exhausted():
            break
        fi = random_walk_path(net, sn, dn, allowed_modes, rng=rng)
        if fi is None:
            continue
        fi_obj = _eval(net, fi, counter)

        for it_idx in range(it):                           # boucle IT
            if counter is not None and counter.exhausted():
                break
            for _ in range(nn):                            # For j = 0 to NN
                if counter is not None and counter.exhausted():
                    break

                if controller is not None:
                    state = controller.build_state(net, fi_obj, counter, it_idx, it)
                    action = controller.choose(state)
                    use_width = (action == "width")
                else:
                    use_width = r < rng.random()           # cf. flowchart : R < rand -> largeur

                v = (neighborhood_width(net, fi, allowed_modes, rng=rng) if use_width
                     else neighborhood_depth(net, fi, rng=rng))
                if v is None:
                    if controller is not None:
                        # pas de voisin généré : pas de récompense, mais on
                        # notifie quand même l'agent pour ne pas biaiser ses
                        # statistiques de succès par omission silencieuse.
                        next_state = controller.build_state(net, fi_obj, counter, it_idx, it)
                        controller.observe(state, action, 0.0, next_state)
                    continue

                v_obj = _eval(net, v, counter)
                v_dominates_fi = dom_fn(v_obj, fi_obj)
                if v_dominates_fi:
                    fi, fi_obj = v, v_obj

                if controller is not None:
                    reward = (controller.compute_reward(v_dominates_fi, v_obj, fi_obj)
                              if hasattr(controller, "compute_reward")
                              else (1.0 if v_dominates_fi else 0.0))
                    next_state = controller.build_state(net, fi_obj, counter, it_idx, it)
                    controller.observe(state, action, reward, next_state)

        non_dom.append((fi, fi_obj))

    if not non_dom:
        return [], []

    objs = [o for _, o in non_dom]
    keep = non_dominated_indices(objs)
    return [non_dom[k][0] for k in keep], [non_dom[k][1] for k in keep]
