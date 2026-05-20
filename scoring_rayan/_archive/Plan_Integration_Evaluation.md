# Plan d'integration — Scoring de confiance des modeles IA

## Contexte

Conformite AI Act (classification haut risque). Chaque modele du pipeline doit produire un **score de confiance/qualite** visible sur l'interface et **log** pour audit. Les thresholds (seuils d'alerte) seront definis en amont via des benchmarks.

---

## Approche generale

1. Une classe `ConfidenceScore` porte le score + metadata flexible
2. `AbstractModel` expose une methode `compute_confidence()` que chaque modele surcharge
3. **`ModelManager.infer()`** appelle `compute_confidence()` apres chaque inference et stocke le resultat sur `instance.confidence` — point d'entree centralise, garanti pour tous les modeles
4. `infer()` continue de retourner exactement le meme type qu'avant — zero breaking change
5. La **tache pipeline** (DataIngestion, OCR, etc.) lit `model_manager.last_confidence(model_id)` apres `infer()` et le persiste en base dans la meme transaction DB que le document
6. Les scores sont remontes au frontend via les requetes existantes du dashboard
7. Les modeles pas encore implementes continuent de fonctionner grace a un `try/except` (rollout progressif)

> **Pourquoi dans `ModelManager.infer()` et non dans `@track_inference` ?**
> Le decorateur sur `AbstractModel.infer()` ne se propage pas aux overrides des sous-classes. Quand `ModelManager.infer()` appelle `instance.infer()`, c'est la methode concrete (ex: `LanguageDetection.infer()`) qui s'execute — le wrapper du decorateur n'est jamais declenche. `ModelManager.infer()` est le seul point d'entree centralise garanti.

---

## Exemples de metadata par modele

| Modele | Exemples de metadata |
|--------|---------------------|
| **LanguageDetection** | `{"detected_lang": "fra_Latn", "top_k": [{"lang": "fra_Latn", "score": 0.98}, {"lang": "eng_Latn", "score": 0.01}]}` |
| **TranslationModel** | `{"beam_score": -1.23, "num_chunks": 4, "src_lang": "fra_Latn", "tgt_lang": "eng_Latn"}` |
| **RerankerModel** | `{"top1_raw_score": 3.72, "num_candidates": 20, "rank_method": "smart"}` |

---

## Modifications par fichier

### 1. `DataPipelines/AbstractModel.py`

```python
# <<< NOUVEAU : classe portant le score + metadata
class ConfidenceScore:
    """Score de confiance + metadata variable pour les logs et l'audit AI Act."""
    def __init__(self, score: float, metadata: dict = None):
        self.score = score           # Score entre 0.0 et 1.0
        self.metadata = metadata or {}

# track_inference : inchange (tracking stats uniquement)

class AbstractModel(ABC):
    confidence: ConfidenceScore = None  # <<< NOUVEAU : mis a jour par ModelManager apres chaque infer()

    # <<< NOUVEAU : a surcharger dans chaque sous-classe
    def compute_confidence(self, input_data, output_data, infer_kwargs=None) -> ConfidenceScore:
        raise NotImplementedError
```

---

### 2. `DataPipelines/LanguageDetection.py`

`infer()` passe de `k=1` a `k=5` (cout FastText negligeable) et cache les predictions dans `self._last_predictions`. `compute_confidence()` les reutilise — un seul appel `detect()` au total.

```python
from DataPipelines.AbstractModel import AbstractModel, ConfidenceScore

class LanguageDetection(AbstractModel):

    # MODIFIE : k=1 → k=5, cache des predictions
    def infer(self, text):
        self._last_predictions = self.language_detection.detect(
            (text[:700] + text[1500:2500]).replace("\n", " "), k=5
        )
        if len(self._last_predictions) > 0:
            return self._last_predictions[0]['lang'], self._last_predictions[0]['score']
        return "eng_Latn", 0.0

    # <<< NOUVEAU
    def compute_confidence(self, input_data, output_data, infer_kwargs=None) -> ConfidenceScore:
        lang, score = output_data
        return ConfidenceScore(
            score=score,
            metadata={
                "detected_lang": lang,
                "top_k": self._last_predictions
            }
        )
```

---

### 3. `DataPipelines/ModelManager.py`

`infer()` appelle `compute_confidence()` apres chaque inference et stocke le resultat sur l'instance.
Ajout de `last_confidence()` pour que les taches pipeline recuperent le score sans exposer l'instance brute.

```python
def infer(self, model_id, *args, **kwargs):
    if not self.is_model_loaded(model_id):
        raise RuntimeError(f"The required model {model_id} is not loaded/available in memory!")

    self.update_inference_stats(model_id)
    instance = self._models[model_id].instance
    result = instance.infer(*args, **kwargs)

    # <<< NOUVEAU : scoring de confiance centralise
    try:
        confidence = instance.compute_confidence(input_data=args, output_data=result, infer_kwargs=kwargs)
        confidence.score = max(0.0, min(1.0, confidence.score))  # clamp [0, 1]
        instance.confidence = confidence
    except NotImplementedError:
        pass  # modele pas encore implemente → on skip
    except Exception as e:
        self.logger.warning(f"Confidence scoring failed for {model_id}: {e}")
    # <<< FIN NOUVEAU

    return result

# <<< NOUVEAU : API pour lire le dernier score sans exposer l'instance
def last_confidence(self, model_id):
    """Retourne le dernier ConfidenceScore calcule pour un modele, ou None."""
    ref = self._models.get(model_id)
    return ref.instance.confidence if ref else None
```

---

### 4. `DataPipelines/DataIngestion.py`

```python
# EXISTANT (inchange)
detected_language, confidence = model_manager.infer(languageDetection, file_content["text"][0:3000])
file_content["metadata"]["language"] = detected_language
file_content["metadata"]["language_confidence"] = confidence

# <<< NOUVEAU : persistance + log du ConfidenceScore complet
lang_confidence = model_manager.last_confidence(languageDetection)
if lang_confidence:
    file_content["metadata"]["confidence_language_detection"] = json.dumps({
        "score": lang_confidence.score,
        "metadata": lang_confidence.metadata
    })
    ModelManager.get_instance().logger.info(
        f"[CONFIDENCE] model={languageDetection} score={lang_confidence.score:.4f} "
        f"metadata={lang_confidence.metadata}"
    )
# <<< FIN NOUVEAU
```

---

## Format de log (audit AI Act)

```
[CONFIDENCE] model=DataPipelines.LanguageDetection.LanguageDetection score=0.9834 metadata={'detected_lang': 'fra_Latn', 'top_k': [{'lang': 'fra_Latn', 'score': 0.98}, ...]}
```

---

## Remontee au client (dashboard)

Les scores sont stockes comme proprietes JSON sur les noeuds Document/Chunk. Pour les afficher :

1. Modifier `get_docs_for_case()` dans `provider_arcadedb.py` pour inclure les proprietes `confidence_*`
2. L'endpoint `/api/cases/{case_id}/dashboard` les remonte automatiquement au frontend
3. Le frontend affiche les scores par traitement pour chaque document
