"""
Jeu d'instances de TEST FINAL (Phase 2) — distinct des instances de
CALIBRATION (`model/calibration.py::build_tuning_instances`, seeds
101–104, tailles 12–25 nœuds) pour éviter le surapprentissage des
paramètres sur les données d'évaluation.


Chaque instance a une paire (origine, destination) fixée explicitement
(plutôt que dérivée dynamiquement de tri de coordonnées comme dans les
scripts d'exemple de la Phase 0/1) : reproductibilité stricte requise pour
la comparaison finale — deux exécutions de `build_final_instances()` sur
la même machine doivent produire EXACTEMENT le même trajet à chaque fois,
indépendamment de tout futur changement de `generate_network`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .entities import Mode
from .network import MultimodalNetwork
from .generator import generate_network

ALL_MODES = {Mode.WALK, Mode.BIKE, Mode.CAR, Mode.BUS, Mode.TRAM, Mode.METRO}


@dataclass
class BenchmarkInstance:
    name: str
    category: str          # "small" | "medium" | "large" | "real"
    n: int
    nb: int
    nt: int
    nm: int
    seed: int
    population_size: int   # taille de population commune à TOUS les algos sur cette instance
    budget: int            # budget d'évaluations commun à TOUS les algos sur cette instance

    def build(self) -> MultimodalNetwork:
        return generate_network(self.n, nb=self.nb, nt=self.nt, nm=self.nm, seed=self.seed)

    def endpoints(self, net: MultimodalNetwork) -> Tuple[str, str]:
        """Origine/destination : les deux nœuds les plus éloignés selon la
        diagonale (x+y) de la zone — heuristique simple garantissant un
        trajet non trivial, IDENTIQUE à celle déjà utilisée dans
        examples/test_all_algorithms.py et test_continuous_algorithms.py
        (Phase 0/1), reprise ici pour rester comparable. Déterministe car
        `generate_network` est déterministe à seed fixée."""
        nodes_sorted = sorted(net.nodes.values(), key=lambda nd: nd.coord.x + nd.coord.y)
        return nodes_sorted[0].id, nodes_sorted[-1].id


# ------------------------------------------------------------------ #
# Règle de dimensionnement (à documenter telle quelle dans l'article) :
#   population_size croît avec la taille du réseau (pratique standard EA :
#   plus l'espace de recherche est grand, plus la population doit être
#   large pour le couvrir) ;
#   budget = 300 x population_size pour TOUTES les instances -- multiplicateur
#   FIXE, choisi pour qu'il corresponde à peu près à 300 générations pour un
#   algorithme à N enfants/génération (NSGA-II/SPEA2/MOEA-D après la
#   correction de la Phase 0), tout en restant simple à justifier et à
#   reproduire. Cette règle simple est documentée comme un choix assumé,
#   pas calibrée finement -- une calibration du budget lui-même sortirait
#   du cadre de cette étude.
# ------------------------------------------------------------------ #
_BUDGET_MULTIPLIER = 300


def _instance(name: str, category: str, n: int, nb: int, nt: int, nm: int,
              seed: int, population_size: int) -> BenchmarkInstance:
    return BenchmarkInstance(name, category, n, nb, nt, nm, seed,
                              population_size, budget=_BUDGET_MULTIPLIER * population_size)


def build_final_instances() -> List[BenchmarkInstance]:
    """9 instances de test final : 3 par catégorie de taille, densités de
    lignes TC variées à taille comparable (pour ne pas confondre effet de
    taille et effet de densité modale). Seeds >= 201, disjointes des seeds
    de calibration (101-104) ET des seeds utilisées dans les scripts
    d'exemple des Phases 0/1 (seed=7 pour le réseau, seed=1 pour les runs
    d'algorithmes -- ceux-là restent des démos, pas le jeu de test)."""
    return [
        # --- petit (comparable en taille aux instances de calibration,
        #     sert de continuité/sanity-check avec la Phase 3) ---
        _instance("final-S1", "small", n=20, nb=2, nt=1, nm=0, seed=201, population_size=30),
        _instance("final-S2", "small", n=22, nb=1, nt=1, nm=1, seed=202, population_size=30),
        _instance("final-S3", "small", n=25, nb=2, nt=2, nm=0, seed=203, population_size=30),

        # --- moyen (régime "réaliste" d'un quartier/petite ville) ---
        _instance("final-M1", "medium", n=45, nb=3, nt=1, nm=1, seed=211, population_size=50),
        _instance("final-M2", "medium", n=50, nb=2, nt=2, nm=1, seed=212, population_size=50),
        _instance("final-M3", "medium", n=55, nb=4, nt=2, nm=0, seed=213, population_size=50),

        # --- grand (stress-test de scalabilité, argument différenciant Q1/Q2) ---
        _instance("final-L1", "large", n=90, nb=5, nt=2, nm=1, seed=221, population_size=80),
        _instance("final-L2", "large", n=100, nb=4, nt=3, nm=1, seed=222, population_size=80),
        _instance("final-L3", "large", n=120, nb=6, nt=3, nm=2, seed=223, population_size=80),

        # "real" : réservé au réseau de Lyon (TCL, cf. ETASR 2025) -- à
        # ajouter ici sous forme d'une BenchmarkInstance construite depuis
        # des données réelles plutôt que generate_network(), dès que ces
        # données sont reconstituées.
    ]


def instances_by_category(category: str) -> List[BenchmarkInstance]:
    return [i for i in build_final_instances() if i.category == category]
