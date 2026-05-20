# Plan d'intégration — ConfidenceScore sur le Reranker

## Contexte

Le Reranker (BAAI/bge-reranker-v2-m3) est le modèle le plus directement lié à la qualité des réponses LLM : il sélectionne les chunks envoyés au LLM depuis un pool de candidats retriévés par embeddings. Son score de confiance est donc un signal fort sur la **qualité du contexte** soumis au LLM.

Le pattern `ConfidenceScore` est déjà en place (AbstractModel, ModelManager, LanguageDetection, Traduction). L'intégration sur le Reranker suit exactement le même schéma.

**Difficulté principale** : `rerank_indexes()` retourne uniquement des indices (`List[int]`) et ne expose pas les scores calculés en interne. Il faut d'abord les stocker sur l'instance.

---

## Signal de confiance

Les scores raw du CrossEncoder (range ~[-10, +10]) sont normalisés via `sigmoid()` (déjà présente dans `reranker.py`) pour obtenir [0, 1].

```
score = 0.6 × top1_score_norm + 0.4 × mean_score_norm
```

- **top1** : meilleur chunk sélectionné — signal de présence d'une réponse pertinente
- **mean** : densité globale du contexte soumis au LLM

---

## Modifications par fichier

### 1. `LargeLanguageHandler/reranker.py` — Exposer les scores sélectionnés

Stocker les scores finaux **avant chaque `return`** dans `rerank_indexes()`. Initialiser à vide en cas d'early return.

```python
def rerank_indexes(self, context, candidates, rank_method='smart', rank_top=5):
    # Early returns
    if len(candidates) == 1:
        self._last_selected_scores = []
        self._last_num_candidates = 1
        self._last_rank_method = rank_method
        return [0]
    elif len(candidates) == 0:
        self._last_selected_scores = []
        self._last_num_candidates = 0
        self._last_rank_method = rank_method
        return []

    # ... (logique existante inchangée jusqu'à la fin) ...

    # Path 'top' — juste avant return, après indexes[:rank_top] :
    self._last_num_candidates = len(candidates)
    self._last_selected_scores = list(scores[:rank_top])  # scores triés desc
    self._last_rank_method = rank_method
    return [i for i in indexes if ...]

    # Path 'kmeans'/'top_kmeans' — juste avant return, après scores_new finaux :
    self._last_num_candidates = len(candidates)
    self._last_selected_scores = list(scores_new)
    self._last_rank_method = rank_method
    return indexes
```

---

### 2. `DataPipelines/Reranker.py` — Implémenter `compute_confidence()`

```python
from DataPipelines.AbstractModel import AbstractModel, ConfidenceScore
from LargeLanguageHandler.reranker import Reranker, ModelMode, sigmoid

class RerankerModel(AbstractModel):

    # ... (inchangé) ...

    def compute_confidence(self, input_data, output_data, infer_kwargs=None) -> ConfidenceScore:
        selected_scores_raw = getattr(self.instance, '_last_selected_scores', [])
        num_in = getattr(self.instance, '_last_num_candidates', 0)
        rank_method = getattr(self.instance, '_last_rank_method', None)

        if not selected_scores_raw:
            return ConfidenceScore(score=0.0, metadata={
                "num_candidates_in": num_in,
                "num_candidates_out": 0,
                "rank_method": rank_method,
            })

        norm_scores = [sigmoid(s) for s in selected_scores_raw]
        top1  = norm_scores[0]
        mean  = sum(norm_scores) / len(norm_scores)
        min_s = min(norm_scores)
        score = 0.6 * top1 + 0.4 * mean

        return ConfidenceScore(
            score=score,
            metadata={
                "num_candidates_in":  num_in,
                "num_candidates_out": len(norm_scores),
                "top1_score":  round(top1,  4),
                "mean_score":  round(mean,  4),
                "min_score":   round(min_s, 4),
                "rank_method": rank_method,
            }
        )
```

---

### 3. `LargeLanguageHandler/LLM_BACKENDS/base.py` — Log après appel reranker

Ajouter juste après le bloc reranker existant (ligne ~880) :

```python
reranker_conf = ModelManager.get_instance().last_confidence("DataPipelines.Reranker.RerankerModel")
if reranker_conf:
    ModelManager.get_instance().logger.info(
        f"[CONFIDENCE] model=DataPipelines.Reranker.RerankerModel "
        f"score={reranker_conf.score:.4f} metadata={reranker_conf.metadata}"
    )
```

> Pas de stockage en nœud graph à ce stade — le reranker intervient pendant le Q&A et non lors de l'ingestion. Le log suffit pour l'audit AI Act.

---

## Format de log

```
[CONFIDENCE] model=DataPipelines.Reranker.RerankerModel score=0.7823 metadata={'num_candidates_in': 100, 'num_candidates_out': 25, 'top1_score': 0.9341, 'mean_score': 0.5612, 'min_score': 0.1042, 'rank_method': 'top_kmeans'}
```

---

## Vérification

1. Poser une question sur un cas avec `use_reranker = True`
2. Vérifier dans les logs la ligne `[CONFIDENCE] model=DataPipelines.Reranker.RerankerModel`
3. Contrôler que `score ∈ [0, 1]` et `num_candidates_out < num_candidates_in`
4. Tester les deux chemins : rank_method `top` et `kmeans`/`top_kmeans`
