# Benchmark Traduction — Mémo de reprise

*Dernière mise à jour : 6 mars 2026 — branche `feature/evaluation-modele`*

---

## Contexte

Comparaison de modèles de traduction pour remplacer ou compléter NLLB-600M dans le pipeline idextend.
Dataset : **FLORES+ devtest** (200 phrases par paire de langues).
Métriques : BLEU, chrF++, vitesse (ms/phrase).
8 paires de langues : FR/EN/AR MSA/AR Levantin (dans les deux sens).

## Fichiers

| Fichier | Rôle |
|---------|------|
| `bench_nllb_vs_qwen.py` | Script principal du benchmark. Wrappers pour chaque modèle, CSV en mode append. Usage : `python bench_nllb_vs_qwen.py --models nllb qwen-4b --n_samples 200` |
| `benchmark_nllb_vs_qwen.csv` | Résultats détaillés (8000 lignes) — une ligne par phrase traduite, contient source/reference/hypothesis/time_ms |
| `benchmark_nllb_vs_qwen_summary.csv` | Résultats agrégés par modèle×paire (BLEU, chrF++, avg_ms, total_s) |
| `benchmark.md` | Notes sur le score de confiance NLLB et AI Act (pas lié au benchmark comparatif) |
| `bench_translation.py` | Benchmark du `TranslationModel.infer()` du repo (code réel, pas FLORES) |
| `bench_translation2.py` | Test de confiance avec phrases easy/hard sur le TranslationModel du repo |
| `Evaluation_modele_Rayan.md` | Notes de la réunion du 16/02 — périmètre évaluation, exigences AI Act |

## Résultats confirmés (200 phrases)

| Paire | NLLB-600M | Qwen-0.8B | Qwen-2B | TGemma-4B | Qwen3.5-4B | Gagnant |
|-------|-----------|-----------|---------|-----------|------------|---------|
| FR→EN | 41.0 | 33.1 | 37.8 | 37.9 | **41.2** | Qwen3.5 |
| EN→FR | **46.3** | 31.6 | 38.9 | 40.4 | 44.7 | NLLB |
| FR→AR | 13.9 | 4.7 | 10.0 | **14.5** | 14.5 | TGemma ≈ Qwen3.5 |
| AR→FR | **29.7** | 12.8 | 20.5 | 26.4 | 29.3 | NLLB |
| FR→AR(Lev) | 7.0 | 3.7 | 6.6 | **8.8** | 7.9 | TGemma |
| AR(Lev)→FR | **28.5** | 11.3 | 19.2 | 23.3 | 27.2 | NLLB |
| EN→AR | **22.0** | 8.4 | 14.6 | 17.9 | 21.7 | NLLB |
| AR→EN | **37.3** | 21.2 | 29.0 | 30.8 | 35.0 | NLLB |

**Vitesse (ms/phrase) :** NLLB ~130-180 | Qwen-0.8B ~470-760 | Qwen-2B ~490-670 | TGemma ~620-940 | Qwen3.5-4B ~660-900

## Conclusions à cette date

1. **NLLB-600M reste le meilleur** : gagne 5-6/8 directions, 4-5x plus rapide que les LLM
2. **Qwen3.5-4B est le meilleur LLM testé** : seul à battre NLLB (sur FR→EN), écarts très serrés partout ailleurs
3. **TGemma-4B** gagne vers l'arabe (FR→AR, FR→AR Lev) mais perd sur le reste
4. **Les résultats à 5 phrases sont trompeurs** : Qwen3.5-4B affichait BLEU 55 à 5 phrases, retombe à 41 à 200
5. Scaling loi confirmée : 0.8B < 2B < 4B, chaque palier gagne ~5-8 points BLEU

## État du CSV — attention aux doublons

Le summary CSV contient **deux runs de Qwen3.5-4B** :
- Lignes 34-41 : run à **5 phrases** (n=5) — résultats gonflés, à ignorer
- Lignes 42-49 : run à **195 phrases** (n=195) — résultats fiables mais sur 195 au lieu de 200

Le CSV détaillé contient 1600 lignes pour qwen3.5-4B (40 du run 5 + 1560 du run 195).
Les 5 premières phrases sont donc en double. Pour un CSV propre, il faudrait supprimer les lignes du run à 5.

## Modèles ajoutés au script mais PAS encore benchmarkés

- `qwen-9b` (Qwen3.5-9B) — plus gros, devrait être meilleur mais plus lent
- `qwen-27b-q4` (Qwen3.5-27B en INT4 via bitsandbytes) — test de la quantification
- `aya-8b` (CohereForAI/aya-23-8B) — modèle multilingue spécialisé, accès HuggingFace accordé mais jamais téléchargé (problèmes réseau SSL + manque protobuf)

## Problèmes connus

- **Réseau HuggingFace instable** : SSL reset fréquents, les modèles en cache fonctionnent mais les nouveaux downloads échouent
- **Aya-23-8B** : accès gated accordé, token HF configuré, mais download jamais abouti
- **CSV en mode append** : risque de doublons si on relance un modèle déjà benchmarké — utiliser `--output_csv` différent ou nettoyer avant

## Prochaines étapes possibles

1. **Nettoyer le CSV** : supprimer le run à 5 phrases de Qwen3.5-4B (ou relancer un run propre à 200 avec `--output_csv benchmark_v2.csv`)
2. **Benchmarker Qwen-9B et Qwen-27B-Q4** : voir si le scaling continue ou si on atteint un plateau
3. **Tester Aya-23-8B** : résoudre le problème réseau/protobuf et lancer le benchmark
4. **Décider** : NLLB reste en prod (meilleur rapport qualité/vitesse), mais Qwen3.5-4B pourrait servir de fallback ou pour des langues non supportées par NLLB
