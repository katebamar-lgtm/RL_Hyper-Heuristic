"""
Instances de DIAGNOSTIC -- un premier contrôle rapide de généralisation
(entraînement vs test), à ne PAS confondre avec le jeu de test FINAL
officiel (`model/benchmark_instances.py::build_final_instances()`, seeds
201-223) qui doit rester "vierge" jusqu'à la campagne de comparaison
définitive (protocole expérimental §4, §9 -- éviter le "garden of forking
paths" : ne regarder le jeu de test qu'une fois le développement/les
ablations terminés).

Seeds 701/711/721 : vérifiées disjointes de toutes les seeds déjà en
usage dans le dépôt (7, 1, 101-104, 201-223, 301/302/311/312 [calibration
medium/large], 601-622 [entraînement RL]).

Tailles volontairement DIFFÉRENTES de build_rl_training_instances() (601-622)
ET de build_final_instances() (201-223) : ce diagnostic teste la
généralisation à des tailles/densités jamais vues ni à l'entraînement ni
dans le jeu de test officiel -- un contrôle plus strict qu'un simple
tirage aléatoire à l'intérieur de la distribution d'entraînement.
"""

from __future__ import annotations
from typing import List

from .train_instances import TrainInstance


def build_diagnostic_instances() -> List[TrainInstance]:
    """3 instances (une par catégorie), seeds 701/711/721."""
    return [
        TrainInstance("diag-S", "small", n=30, nb=3, nt=1, nm=1, seed=701),
        TrainInstance("diag-M", "medium", n=65, nb=3, nt=2, nm=1, seed=711),
        TrainInstance("diag-L", "large", n=100, nb=5, nt=2, nm=2, seed=721),
    ]
