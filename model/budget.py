"""
Compteur d'évaluations de fitness, partagé entre tous les algorithmes du projet.

Essentiel pour la calibration : deux configurations avec des NN/IT
différents ne font PAS le même nombre d'appels à evaluate_route() par
génération. Pour rester équitable, on limite chaque exécution par un
BUDGET TOTAL d'évaluations (plutôt qu'un nombre de générations), identique
pour toutes les configurations comparées.
"""

from __future__ import annotations
from typing import Optional
from .network import MultimodalNetwork, Connection


class EvalCounter:
    """Compte TOUS les appels à evaluate_route() d'un run, y compris ceux
    internes à OMF (voisins testés en recherche locale). C'est essentiel :
    sans ça, certains algorithmes bénéficieraient d'évaluations "gratuites" invisibles au
    budget partagé avec NSGA-II/SPEA2/MOEA/D, ce qui biaiserait toute
    comparaison en sa faveur (ou sa défaveur) selon (IT, NN).

    budget=None -> pas de limite (comportement historique, rétrocompatible :
    tous les runs existants sans EvalCounter continuent de fonctionner à
    l'identique).
    """

    def __init__(self, budget: Optional[int] = None):
        self.count = 0
        self.budget = budget

    def evaluate(self, net: MultimodalNetwork, path) -> tuple:
        self.count += 1
        return net.evaluate_route(path).as_tuple()

    def exhausted(self) -> bool:
        return self.budget is not None and self.count >= self.budget

    def remaining(self) -> Optional[int]:
        """None si pas de budget fixé, sinon nombre d'évaluations restantes
        (peut être négatif si le dernier lot a dépassé le budget d'une
        petite marge — les boucles appelantes vérifient `exhausted()` entre
        deux évaluations, pas pendant, donc un léger dépassement ponctuel
        est possible et acceptable)."""
        return None if self.budget is None else self.budget - self.count

    def __repr__(self) -> str:
        b = "∞" if self.budget is None else self.budget
        return f"EvalCounter({self.count}/{b})"
