# LLM — Classification du biais politique

Tester si un LLM est capable d'identifier la sensibilité politique d'un article de presse (droite / gauche / neutre), et évaluer la fiabilité de cette classification.

## Structure

```
data/raw/          # Articles bruts (texte + métadonnées source)
data/processed/    # Dataset CSV annoté (article_id, texte, source, etiquette_ground_truth, theme)
notebooks/         # Inférence LLM + analyse des résultats
src/               # Scripts utilitaires
```

## Protocole

- Input : article de presse en texte brut, sans indication de source
- Tâche LLM : classer l'article (droite / gauche / neutre) et justifier
- Dataset : 20-40 articles couvrant les mêmes événements depuis des sources aux orientations connues (ex. Le Figaro / Libération, Fox News / The Guardian)
- Thèmes : tensions sociales, faits criminels, politique policière, immigration

## Modèles testés

Qwen3-4b, Qwen3.5-4B, Qwen3.6-27B

## Métriques

- Accuracy par modèle et par thème
- Analyse qualitative des justifications (vocabulaire, faits cités, cadrage narratif)
- Question bonus : après lecture, le LLM s'aligne-t-il avec la sensibilité de l'article quand on lui demande son opinion ?

## Spec complète

Voir [`../LLM_Bias/data/results/spec_classification_biais_politique.md`](../LLM_Bias/data/results/spec_classification_biais_politique.md)

> **Note :** expérience non encore implémentée — les dossiers `data/`, `notebooks/`, `src/` sont vides.
