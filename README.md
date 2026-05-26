# Mémoire — Biais des LLMs

Dépôt des expériences pour le mémoire sur la détection et la mesure des biais dans les modèles de langage.

## Expériences

| Dossier | Objet |
|---|---|
| [`IAT_replication/`](IAT_replication/) | Réplication du test IAT sur LLMs — mesure de biais implicites (race, genre, âge…) via D-scores |
| [`LLM_Bias/`](LLM_Bias/) | Biais en contexte d'embauche — évaluation de CV (condition single vs. comparative) |
| [`llm_bridage_modeles/`](llm_bridage_modeles/) | Bridage des LLMs sur contenu sensible — refus, restitution partielle ou complète |
| [`llm_classification_biais_politique/`](llm_classification_biais_politique/) | Classification du biais politique d'articles de presse (gauche / droite / neutre) |
| [`logprobs_evaluation/`](logprobs_evaluation/) | Calibration des log-probs — corrélation confiance / qualité de réponse sur articles |

## Modèles utilisés

Tous les tests utilisent des modèles **open-weight** déployés localement via vLLM (`temperature=0`). Les modèles varient selon l'expérience — voir le README de chaque dossier.
