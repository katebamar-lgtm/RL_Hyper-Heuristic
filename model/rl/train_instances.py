"""
Jeu d'instances d'ENTRAÎNEMENT de l'agent RL — strictement disjoint des
seeds déjà utilisées ailleurs dans le dépôt :

  - calibration classique (model/calibration.py)      : seeds 101-104
  - test final synthétique (model/benchmark_instances) : seeds 201-223
  - Lyon (data/lyon)                                   : réseau réel, hors seed
  - ENTRAÎNEMENT RL (ce module)                        : seeds >= 301

Règle impérative reprise du protocole expérimental : aucune de ces seeds
ne doit jamais recouvrir celles ci-dessus, ni être réutilisée comme
instance de test -- c'est la condition qui rend l'évaluation de
généralisation (RQ1 du protocole) valide plutôt que circulaire.

Couvre les trois catégories de taille (comme build_final_instances()),
avec des tailles/densités DIFFÉRENTES de celles du jeu de test, pour que
l'agent apprenne une politique qui généralise plutôt qu'il ne mémorise la
distribution exacte du jeu de test.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from ..entities import Mode
from ..network import MultimodalNetwork
from ..generator import generate_network
from ..benchmark_instances import BenchmarkInstance  # réutilisé seulement pour .endpoints()

ALL_MODES = {Mode.WALK, Mode.BIKE, Mode.CAR, Mode.BUS, Mode.TRAM, Mode.METRO}

# Instance "vide" utilisée uniquement pour appeler .endpoints(net), qui ne
# dépend d'aucun champ de l'instance elle-même (cf. benchmark_instances.py).
_ENDPOINT_PICKER = BenchmarkInstance("_", "_", 0, 0, 0, 0, 0, 0, 0)


@dataclass
class TrainInstance:
    name: str
    category: str      # "small" | "medium" | "large"
    n: int
    nb: int
    nt: int
    nm: int
    seed: int

    def build(self) -> MultimodalNetwork:
        return generate_network(self.n, nb=self.nb, nt=self.nt, nm=self.nm, seed=self.seed)

    def endpoints(self, net: MultimodalNetwork) -> Tuple[str, str]:
        return _ENDPOINT_PICKER.endpoints(net)


def build_rl_training_instances() -> List[TrainInstance]:
    """6 instances d'entraînement (2 par catégorie), seeds 601-622.

    ATTENTION (corrigé) : une première version de ce fichier utilisait les
    seeds 301/302/311/312/321/322, qui entrent en collision directe avec
    `model/calibration.py::build_tuning_instances_medium_large()` (seeds
    301, 302, 311, 312 -- calibration multi-échelle de NSGA-II/
    SPEA2/MOEA-D, Phase 4-bis). Corrigé en 601-622, un bloc de seeds
    n'apparaissant nulle part ailleurs dans le dépôt (vérifié par recherche
    exhaustive : 7, 1 (démos Phase 0/1), 101-104 (calibration small),
    301/302/311/312 (calibration medium/large), 201-223 (test final)).
    Tailles volontairement différentes de celles de build_final_instances()
    (20/22/25, 45/50/55, 90/100/120) pour éviter tout chevauchement, même
    accidentel, avec le jeu de test final."""
    return [
        TrainInstance("train-S1", "small", n=18, nb=2, nt=1, nm=0, seed=601),
        TrainInstance("train-S2", "small", n=24, nb=1, nt=2, nm=1, seed=602),

        TrainInstance("train-M1", "medium", n=40, nb=3, nt=1, nm=1, seed=611),
        TrainInstance("train-M2", "medium", n=58, nb=2, nt=2, nm=1, seed=612),

        TrainInstance("train-L1", "large", n=85, nb=4, nt=2, nm=1, seed=621),
        TrainInstance("train-L2", "large", n=110, nb=5, nt=3, nm=2, seed=622),
    ]


def max_training_n_nodes() -> int:
    """n_nodes_ref pour ContextualBandit -- la plus grande instance
    d'ENTRAÎNEMENT, jamais recalculée à partir d'une instance de test."""
    return max(inst.n for inst in build_rl_training_instances())
