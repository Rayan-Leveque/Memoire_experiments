# IAT Replication — Biais implicites des LLMs

Mesure des biais implicites dans les LLMs via une adaptation du test IAT (Implicit Association Test). Les modèles assignent des mots à des groupes démographiques ; un **D-score** est calculé pour quantifier le biais.

5 catégories couvertes : race, genre, âge, santé, religion.

## Structure

```
src/
  cleaning.py        # Pipeline : réponses brutes → D-scores
  iat_benchmark.py   # Inférence : charge les modèles, lance les essais IAT
  config_loader.py   # Gestion de la config YAML
  weat.py            # Métriques WEAT/WEFAT (approche par embeddings)
config/
  models_config.yml  # Config matérielle, modèles, paramètres d'évaluation
data/
  iat_stimuli.csv    # Stimuli IAT (250 lignes)
  results/           # CSV bruts par modèle
output/              # Résultats nettoyés + figures
notebooks/           # Exploration et analyse
```

## Usage

```bash
# Inférence (génère data/results/*.csv)
python src/iat_benchmark.py

# Nettoyage et calcul des D-scores
python src/cleaning.py --input data/results/ --output output/result_implicit.csv
```

## Interprétation du D-score

- `D > 0` : biais contre le groupe stigmatisé
- `D < 0` : biais contre le groupe de référence
- `D = 0` : absence de biais

## Modèles testés

Qwen3-30B, Mistral-Small-3.2-24B, Phi-4, GPT-OSS-20B (via vLLM, `temperature=0`).

## Prérequis

`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `vllm`, `torch`, `pyyaml`, `tqdm`, `sympy`, `openai`
