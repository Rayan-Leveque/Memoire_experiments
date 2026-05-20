# Dossier de documentation — Scoring de confiance IDeXtend

> Conformité AI Act (Annexe III §6 — système haut risque, application de la loi).
> Ce dossier documente scientifiquement le scoring de confiance de chaque composant du pipeline IDeXtend.

---

## Structure

```
scoring_rayan/docs/
├── README.md               ← ce fichier
└── latex/                  ← source unique — fiches composants + build PDF
    ├── Makefile
    ├── main.tex
    ├── preamble.tex
    ├── references.bib
    ├── chapters/           ← une fiche .tex par composant (éditer ici)
    │   ├── 01_ocr.tex
    │   ├── 02_asr_whisper.tex
    │   ├── 03_embeddings.tex
    │   ├── 04_reranker.tex
    │   ├── 05_llm_qwen.tex
    │   ├── 06_translation_nllb.tex
    │   └── 07_detection_langue.tex
    └── build/
        └── main.pdf        ← rapport compilé
```

---

## État d'implémentation des scores

| # | Composant | Modèle | `compute_confidence()` | Validé empiriquement |
|---|---|---|---|---|
| 1 | OCR | QwenVL 2.5-7B | ⬜ À implémenter | ⬜ |
| 2 | ASR (Whisper) | whisper-large-v3 | ⬜ À implémenter | ⬜ |
| 3 | Embeddings | BAAI/bge-m3 | ⬜ À implémenter | ⬜ |
| 4 | Reranker | BAAI/bge-reranker-v2-m3 | ✅ Implémenté | ⬜ |
| 5 | LLM (Q&A) | Qwen3-4B | ⬜ À définir + implémenter | ⬜ |
| 6 | Traduction | NLLB-600M | ✅ Implémenté | ⬜ |
| 7 | Détection langue | FastText LID | ✅ Implémenté | ⬜ |

---

## Générer le PDF

```bash
# Alias disponible (après source ~/.bashrc) :
build-scoring

# Ou manuellement :
cd scoring_rayan/docs/latex && PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH make pdf
# → scoring_rayan/docs/latex/build/main.pdf
```

---

## Workflow

Éditer directement les `.tex` dans `latex/chapters/`, puis lancer `build-scoring`.

---

## Prochaines étapes

1. Constituer les datasets d'évaluation (section 4 de chaque fiche)
2. Implémenter `compute_confidence()` pour OCR, Whisper, Embeddings, LLM
3. Conduire les expériences de validation empirique (Spearman, ECE, AUC-ROC)
4. Définir l'approche de scoring composite LLM (fiche 05)
5. Compléter le benchmark OCR (Marker, Docling vs QwenVL)

---

## Références archives

Les benchmarks et mémos existants sont dans `scoring_rayan/_archive/` :
- `benchmark_traduction_memo.md` — résultats benchmark traduction NLLB vs Qwen
- `Plan_Integration_Evaluation.md` — plan d'intégration du scoring de confiance
- `Plan_Integration_Reranker.md` — plan d'intégration du reranker



Rajouter des exmemples a chaque items
{
    "query" : "xyz",
    "answer" "zyx"
}