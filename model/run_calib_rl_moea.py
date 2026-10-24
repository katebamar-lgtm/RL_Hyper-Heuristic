"""
Script de lancement de la calibration RL-MOEA/D.

Ce script ne modifie pas model/calibration.py.
Il appelle simplement la fonction calibrate_rl_moead()
et sauvegarde le résultat dans :

```
calibration/final_rl_moead.json
```

"""

import json
from pathlib import Path

from model.calibration import calibrate_rl_moead

def main():

 print("\n" + "=" * 70)
 print("CALIBRATION RL-MOEA/D")
 print("=" * 70)

 print("\nProtocole :")
 print("  - Population       : 30")
 print("  - Budget           : 2000 évaluations")
 print("  - Répétitions      : 5")
 print("  - Instances        : 4 instances de calibration")
 print("  - Métrique         : Hypervolume normalisé")
 print("\nÉtape 1 : calibration de neighborhood_size (T)")
 print("Étape 2 : calibration de ucb_c")
 print()

# ---------------------------------------------------------------
# Lancement de la calibration
# ---------------------------------------------------------------

 config, best_hv, ties, results = calibrate_rl_moead(
     verbose=True
 )

# ---------------------------------------------------------------
# Affichage des résultats finaux
# ---------------------------------------------------------------

 print("\n" + "=" * 70)
 print("RESULTATS FINAUX")
 print("=" * 70)

 print("\nMeilleure configuration :")

 for key, value in config.items():
    print(f"  {key}: {value}")

 print(f"\nMeilleur Hypervolume : {best_hv:.10f}") 

 print("\nConfigurations statistiquement équivalentes :")

 for tie in ties:
    print(f"  - {tie}")

# ---------------------------------------------------------------
# Sauvegarde du résultat
# ---------------------------------------------------------------

 project_root = Path(__file__).resolve().parent

 output_dir = project_root / "calibration"
 output_dir.mkdir(exist_ok=True)

 output_file = output_dir / "final_rl_moead.json"

 output = {
    "config": config,
    "hv": best_hv,
    "ties": [str(tie) for tie in ties]
 }

 with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

 print("\n" + "=" * 70)
 print("SAUVEGARDE")
 print("=" * 70)

 print(f"\nRésultats sauvegardés dans :")
 print(output_file)

 print("\nCalibration terminée avec succès.")

if __name__ == "__main__":
    main()