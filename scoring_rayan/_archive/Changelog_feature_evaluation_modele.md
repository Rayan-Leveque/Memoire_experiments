## Commit 1 — `562e6029` — Ajout d'un score de confiance sur la classe abstraite et LanguageDetection

### Fichiers modifiés

**`DataPipelines/AbstractModel.py`**
- Ajout de la classe `ConfidenceScore` : conteneur générique `{ score: float, metadata: dict }`.
- Ajout de l'attribut de classe `confidence: ConfidenceScore = None` sur `AbstractModel`.
- Ajout de la méthode abstraite `compute_confidence(input_data, output_data, infer_kwargs)` que chaque sous-classe doit implémenter.

**`DataPipelines/ModelManager.py`**
- La méthode `infer()` appelle désormais `compute_confidence()` après chaque inférence, clamp le score dans [0, 1], et le stocke dans `instance.confidence`.
- Nouvelle méthode `last_confidence(model_id)` : retourne le dernier `ConfidenceScore` calculé pour un modèle donné.
- Les erreurs de scoring sont catchées et loggées en warning sans bloquer l'inférence.

**`DataPipelines/LanguageDetection.py`**
- Implémentation de `compute_confidence()` : utilise directement le score de fastText (déjà en [0, 1]) sur la langue détectée.
- Stocke également les top-k prédictions dans les métadonnées.
- La méthode `infer()` retourne maintenant `(lang, score)` au lieu de `lang` seul pour permettre à `compute_confidence` d'accéder au score brut.

**`DataPipelines/DataIngestion.py`**
- Récupère le `ConfidenceScore` de la détection de langue via `model_manager.last_confidence()`.
- Stocke la confidence en JSON dans `file_content["metadata"]["confidence_language_detection"]`.
- Log dans le logger principal avec le format `[CONFIDENCE] model=... score=... metadata=...`.

---

## Commit 2 — `9b096d1b` — Evaluation de la traduction

### Fichiers modifiés

**`DataPipelines/TranslationModel.py`**

- Remplacement du `pipeline` HuggingFace par des appels directs à `model.generate()` avec `return_dict_in_generate=True` et `output_scores=True` (nécessaire pour accéder à `sequences_scores`).
- Collecte de métriques par chunk à chaque génération :
  - `seq_score` : log-probabilité normalisée du meilleur beam (beam search score, range ~(-∞, 0))
  - `length_ratio` : ratio tokens_output / tokens_input
  - `input_len` / `output_len` en tokens
- Stockage des métriques dans des attributs d'instance (`_last_chunk_metrics`, `_last_num_chunks`, etc.) pour que `compute_confidence()` puisse y accéder après coup.
- Implémentation de `compute_confidence()` :
  - Convertit les `seq_score` en probabilités via `exp()` → plage [0, 1]
  - Score composite : `0.7 × mean_beam_prob + 0.3 × min_beam_prob` (pénalise les pires chunks)
  - Pénalité outlier : chunks avec `length_ratio < 0.3` ou `> 4.0` (troncature ou génération runaway) réduisent le score de 50% au prorata
  - Métadonnées complètes : stats globales + détail par chunk

**`DataPipelines/TranslationTask.py`**

- Passage de `translatorInstance.infer(...)` à `mm.infer(self.NLLB_MODEL, ...)` pour passer par le `ModelManager` et déclencher automatiquement le calcul de confiance.
- Récupération du `ConfidenceScore` via `mm.last_confidence()` après chaque chunk traduit.
- Stockage en JSON dans `features["confidence_translation"]`.
- Log avec le format `[CONFIDENCE] model=... score=... metadata=...` (hors champ `per_chunk` pour ne pas polluer les logs).

---

---

## Benchmark de validation du score de confiance (`bench_translation2.py`)

### Protocole

- **3 langues sources** : `fra_Latn`, `eng_Latn`, `por_Latn`
- **4 langues cibles** (hors identité) : `eng_Latn`, `fra_Latn`, `por_Latn`, `spa_Latn`, `ita_Latn`
- Métrique collectée : `confidence_score`, `mean_beam_prob`, `min_beam_prob`, `mean_length_ratio`, `num_chunks`, `outlier_chunks`, `text_len`

### Résultats clés

**Par langue cible :**
| lang_tgt | confidence_score | mean_beam_prob |
|----------|-----------------|----------------|
| eng_Latn | 0.5915          | 0.5920         |
| fra_Latn | 0.5992          | 0.5999         |
| ita_Latn | 0.5766          | 0.5773         |
| por_Latn | 0.5719          | 0.5723         |
| spa_Latn | 0.5996          | 0.6003         |

**Par langue source :**
| lang_src | confidence_score | mean_beam_prob |
|----------|-----------------|----------------|
| eng_Latn | 0.5705          | 0.5711         |
| fra_Latn | 0.5718          | 0.5723         |
| por_Latn | 0.6224          | 0.6231         |

### Observations

- **Seuil `< 0.4`** — cas systématiquement sous le seuil : répétitions (`oui oui oui...`), bruit pur (`??? !!!`), mots tronqués (`Les enfa`), fautes massives. La syntaxe cassée et les mélanges de langues restent au-dessus de 0.4 (le modèle hallucine une traduction plausible et reste confiant).
- **Langue cible** — effet faible (~0.028 d'écart entre meilleure et pire cible). `ita_Latn` et `por_Latn` sont structurellement plus faibles en cible (moins de données NLLB). L'effet langue source est également faible.
- **Conclusion seuil** : `0.4` est conservateur (ne rattrape pas les cas de syntaxe cassée), `0.5` serait plus couvrant mais risque des faux positifs sur textes courts légitimes.

---

## Notes techniques
- **`output_scores=True`** est obligatoire pour obtenir `sequences_scores` dans transformers 4.57+. Sans lui, `sequences_scores` est `None`.
- Le scoring ne bloque jamais l'inférence : toute exception dans `compute_confidence()` est catchée dans `ModelManager.infer()`.
