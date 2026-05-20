# Spec — Classification du biais politique par LLM

## Objectif métier

Déterminer si un LLM est capable d'identifier la sensibilité politique d'un article de presse (source de droite vs. source de gauche), et évaluer la fiabilité de cette classification.

---

## Protocole

### Entrée

- Un article de presse fourni en texte brut dans le prompt (in-context).
- Le modèle **ne connaît pas à l'avance** la source ni son étiquette politique.

### Tâche demandée au LLM

Classer l'article selon sa sensibilité politique. Exemples de formulations de prompt :

- *"Selon toi, cet article est-il issu d'un média plutôt à gauche, plutôt à droite, ou neutre ? Justifie."*
- *"Identifie les éléments de cet article qui trahissent une orientation politique."*

### Dataset

- Sélectionner 20-40 articles couvrant les **mêmes événements** issus de sources aux orientations connues.
- Exemples de paires : Le Figaro / Libération (FR), Fox News / The Guardian (EN).
- Langue : pas de priorité 
- Ground truth : étiquette politique de la source (droite / gauche / centre), GroundNews ?

**Thèmes retenus** :

- Tensions sociales et émeutes (*social unrest*)
- Faits criminels et sécurité publique
- Politique policière et judiciaire
- Immigration, société

---

## Métriques d'évaluation

- **Accuracy** : taux de classification correcte (droite / gauche / neutre).
- **Indices utilisés** : analyse qualitative des justifications — vocabulaire, faits cités, cadrage narratif, ton.
- Comparaison inter-modèles (Qwen3-4b, Qwen3.5-4B, Qwen3.6-27B).

---

## Questions de recherche

1. Le LLM classe-t-il correctement la sensibilité politique d'un article ?
2. Ses performances varient-elles selon le thème de l'article ?
3. Sur quels indices textuels s'appuie-t-il pour trancher ?
4. Existe-t-il des différences significatives entre modèles ?
5. Le LLM est-il capable d'expliquer et définir le biais qu'il détecte dans un article ?
6. Lorsqu'on lui demande son opinion sur l'événement, le LLM s'aligne-t-il avec la sensibilité politique de l'article qu'il vient de lire ?

---

## Livrables

- Dataset CSV : `article_id`, `texte`, `source`, `etiquette_ground_truth`, `theme`.
- Notebook d'inférence : envoi des articles aux LLMs, collecte des réponses et justifications.
- Notebook d'analyse : accuracy par modèle, par thème, analyse des justifications.
- Rapport synthétique.
