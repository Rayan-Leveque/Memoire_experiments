# Fiche composant — [Nom du composant]

> Template de documentation du scoring de confiance pour les composants d'IDeXtend.
> À remplir pour chaque brique : OCR, ASR (Whisper), Embeddings, Reranker, LLM (Qwen), etc.
> Cible : conformité AI Act (Art. 9, 13, 15) + traçabilité scientifique pour publication.

---

## 1. Identification

| Champ | Valeur |
|---|---|
| Nom du composant | |
| Version / modèle retenu | |
| Date de rédaction | |
| Auteur(s) | |
| Statut | brouillon / revue / validé |
| Place dans le pipeline IDeXtend | |

## 2. Définition de la fonctionnalité

**Rôle dans le système.**
*Décrire en 3-5 lignes ce que fait ce composant, son entrée, sa sortie, et son impact sur la décision d'enquête.*

**Entrée.**
- Format :
- Source :
- Volume typique :

**Sortie.**
- Format :
- Consommateur en aval :

**Criticité AI Act.**
*Identifier en quoi ce composant contribue au caractère « haut risque » du système (Annexe III §6 — application de la loi). Risques spécifiques d'erreur et conséquences sur l'enquête.*

## 3. Modèles évalués

| Modèle | Version | Source | Inclus dans le benchmark | Retenu |
|---|---|---|---|---|
| | | | oui/non | oui/non |
| | | | | |

**Critères de présélection.** *Pourquoi ces modèles et pas d'autres (open-source, licence compatible Gendarmerie, langue FR, performance publiée, etc.).*

## 4. Protocole d'évaluation

**Dataset d'évaluation.**
- Nom / description :
- Taille :
- Représentativité par rapport aux données de production (préciser le gap éventuel) :
- Annotation : qui, quand, accord inter-annotateurs si pertinent
- Disponibilité (interne, public, restreint) :

**Métriques de performance brute.**
*Métriques utilisées pour comparer les modèles (CER/WER pour OCR, ROUGE/BLEU/BERTScore pour LLM, MRR/nDCG pour retrieval, etc.).*

**Résultats du benchmark.**

| Modèle | Métrique 1 | Métrique 2 | Latence | VRAM |
|---|---|---|---|---|
| | | | | |

**Justification du modèle retenu.** *Argumentaire pondéré : performance, coût, robustesse, licence.*

## 5. Fonction de scoring de confiance

**Définition formelle.**

Notation : soit $x$ l'entrée, $\hat{y}$ la sortie du modèle, $\theta$ ses paramètres.
Le score de confiance est défini par :

$$
s(x, \hat{y}) = \ldots
$$

*Écrire la formule complète. Préciser chaque terme, les agrégations utilisées (mean, min, weighted), les normalisations.*

**Plage de valeurs.** $s \in [\,\ldots\,]$

**Interprétation des seuils.**

| Plage | Interprétation | Action recommandée |
|---|---|---|
| $s \geq \tau_{\text{haut}}$ | Haute confiance | Exploitable sans relecture humaine systématique |
| $\tau_{\text{bas}} \leq s < \tau_{\text{haut}}$ | Confiance moyenne | Relecture recommandée |
| $s < \tau_{\text{bas}}$ | Faible confiance | Relecture humaine obligatoire |

## 6. Justification scientifique de la fonction

**État de l'art.**
*Citer les références qui valident l'approche choisie pour ce type de modèle. Au moins 3-5 références.*

- [Ref 1] —
- [Ref 2] —
- [Ref 3] —

**Alternatives considérées et rejetées.**
*Pourquoi d'autres fonctions de scoring (semantic entropy, P(True), ensemble disagreement, etc.) n'ont pas été retenues — ou si elles ont été retenues en complément.*

**Hypothèses sous-jacentes.**
*Conditions sous lesquelles le score est censé être informatif (ex. : hypothèse de calibration locale, distribution d'entrée proche du training, etc.).*

## 7. Validation empirique du score

> Le cœur du dossier : prouver que le score corrèle avec la qualité réelle de la sortie.

**Protocole.**
*Décrire le protocole : dataset annoté avec qualité ground-truth, binning du score, comparaison.*

**Corrélation score ↔ qualité.**

| Test statistique | Valeur | p-value | Interprétation |
|---|---|---|---|
| Spearman ρ | | | |
| Pearson r | | | |
| Kendall τ | | | |

**Calibration.**

- Expected Calibration Error (ECE) :
- Maximum Calibration Error (MCE) :
- Brier Score :
- Reliability diagram : *(joindre figure)*

**Discrimination (si binarisation pertinente).**

- AUC-ROC :
- AUC-PR :
- Seuils optimaux et matrice de confusion à ces seuils :

**Stress tests.**
*Comportement sur cas dégradés explicites : entrée bruitée, hors distribution, adversarial.*

| Cas | Score moyen attendu | Score moyen observé | Conforme ? |
|---|---|---|---|
| Entrée haute qualité | élevé | | |
| Entrée moyenne | médian | | |
| Entrée dégradée | bas | | |
| Hors distribution | bas ou flag | | |

## 8. Limites et conditions d'usage

**Cas où le score est non fiable.**

**Biais identifiés.**
*Performance différentielle selon des sous-populations (langue, type de document, qualité d'image, locuteur, etc.).*

**Dérive prévisible.**
*Conditions sous lesquelles un recalibrage est nécessaire.*

## 9. Conformité AI Act

| Article / exigence | Comment ce composant y répond |
|---|---|
| Art. 9 — Gestion des risques | |
| Art. 10 — Données et gouvernance | |
| Art. 13 — Transparence | |
| Art. 14 — Contrôle humain | |
| Art. 15 — Exactitude, robustesse, cybersécurité | |

## 10. Monitoring en production

**Métriques de suivi.**

**Conditions de déclenchement d'une revue.**

**Logs et traçabilité.**

## 11. Annexes

- Code de référence (chemin repo) :
- Notebooks d'évaluation :
- Datasets utilisés (chemins) :
- Rapports de benchmark détaillés :
