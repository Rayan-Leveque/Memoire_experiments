# Logprobs Evaluation — Calibration de la confiance des LLMs

Évaluation empirique de la corrélation entre les **log-probabilités** de LLMs et la qualité réelle de leurs réponses sur des articles de presse.

## Hypothèse

Un LLM bien calibré devrait produire des logprobs élevés uniquement sur des questions auxquelles l'article permet de répondre précisément.

## Protocole (4 quadrants)

| | Question adaptée | Question non-adaptée |
|---|---|---|
| **Info précise** | A — logprobs élevés attendus | B — logprobs bas attendus |
| **Info large** | C — logprobs moyens | D — alerte si logprobs élevés |

Les questions sont définies dans des fichiers YAML avec `quadrant`, `answerable` et `human_score` (à annoter manuellement).

## Structure

```
eval_logprobs_qwen.py          # Script principal d'évaluation
questions_logprobs_exemple.yaml  # Questions de test (article Al Jazeera)
questions_logprobs_lemonde_macron.yaml
articles/                      # Articles sources en texte brut
data/                          # Résultats
analyse_logprobs.ipynb         # Analyse et corrélation Spearman
```

## Usage

```bash
python eval_logprobs_qwen.py \
    --questions questions.yaml \
    --articles-dir ./articles \
    --api-url http://localhost:8000/ \
    --model Qwen/Qwen3-4B \
    --output results.csv
```

## Modèles testés

Qwen3-4B, Qwen3.5-(0.8B, 2B, 4B), Qwen3.6-(27B, 35B) — chacun en mode contraint et free.

## Métriques

Corrélation de Spearman entre `mean_logprob` et `human_score` par quadrant.
