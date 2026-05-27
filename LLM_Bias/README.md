# LLM Bias — Biais d'évaluation des LLMs en contexte d'embauche

Pipeline expérimental pour mesurer les biais des LLMs lors de l'évaluation de CV, en contexte français.

## Hypothèse

Lorsqu'un LLM évalue des candidats **séparément**, il tend à favoriser les minorités (sur-correction RLHF). Lorsqu'il est placé en **comparaison directe**, il tend à favoriser la majorité (activation de stéréotypes implicites).

Cette dissociation est testée via deux conditions expérimentales (single vs. comparative) sur des CV dont seuls les marqueurs d'identité varient.

Pour le détail du design expérimental, voir [docs/design.md](docs/design.md).

## Pourquoi des modèles open-source / on-premise ?

Ce pipeline teste exclusivement des modèles **open-weight** déployables localement (vLLM) ou via des APIs cloud open-source (Novita AI). Nous ne testons pas GPT-4, Claude ou Gemini. Ce choix est délibéré et méthodologiquement justifié :

1. **Validité écologique du déploiement.** Les systèmes de pré-sélection automatisée en entreprise sont massivement déployés *on-premise* ou via des APIs open-source, non via des APIs propriétaires coûteuses et instables. Tester ces modèles, c'est auditer les outils réellement utilisés par les DSI et les éditeurs de RH.

2. **Reproductibilité scientifique.** Les poids des modèles open-source sont figés et versionnés. Un reviewer peut relancer exactement le même modèle (mêmes poids, même tokenizer, mêmes hyperparamètres) dans 2 ans. Les modèles closed-source changent de version sans préavis ("GPT-4-turbo" d'aujourd'hui n'est pas celui de demain), rendant toute réplication impossible.

3. **Transparence du comportement.** En local, nous contrôlons réellement `temperature=0`, le format de sortie, et l'absence de post-traitement propriétaire. Les APIs closed-source appliquent des filtres, des re-roll, ou des systèmes de modération non documentés qui contaminent la mesure du biais.

4. **Complémentarité avec la littérature existante.** Bai et al. (2024) et Gallegos et al. (2025) ont déjà documenté le phénomène sur GPT-4 et Claude. Notre contribution n'est pas de répéter ces résultats, mais de tester leur **généralisation à l'écosystème open-source** et à un **contexte linguistique et culturel non anglophone**.

5. **Scalabilité économique.** Tester 8 modèles sur ~1 100 appels chacun coûte quelques euros en cloud open-source contre plusieurs centaines d'euros en API propriétaire. Cela permet des analyses inter-modèles riches (taille, architecture, alignement) impossibles autrement.

## Structure du projet

```
├── config.yml              # Configuration centrale (hardware, modèles, pipeline)
├── run_pipeline.py         # Point d'entrée principal
├── run_both_variants.sh    # Lance single + comparative
├── requirements.txt
├── docs/                   # Documentation et design expérimental
├── data/
│   ├── base_profiles/      # Profils de base générés (JSON)
│   ├── profiles/           # Profils injectés (ethnie × SES)
│   └── results/            # Résultats bruts des expériences
├── src/
│   ├── generation/         # Génération des profils de CV
│   └── evaluation/         # Appels LLM (single, comparative, IAT)
├── utils/                  # Clients LLM, lancement vLLM, parsing
├── logs/                   # Réponses brutes (JSONL)
└── paper/                  # Article LaTeX
```

## Setup

```bash
uv sync --group pipeline --group analysis
```

## Usage

### Novita AI (cloud — `Qwen3.7-max-Novita`, `Mistral-Nemo-Novita`)

`--models` prend les **display names** (pas les model IDs). La clé API doit être exportée manuellement à chaque session.

```bash
export NOVITA_API_KEY=sk-<your_key>
nohup .venv/bin/python run_pipeline.py \
  --no-auto-vllm \
  --step all \
  --models "Qwen3.7-max-Novita,Mistral-Nemo-Novita" \
  > logs/pipeline.log 2>&1 &
tail -f logs/pipeline.log
```

### Local (vLLM — `Gemma-4-31B-it`, etc.)

Le serveur vLLM est démarré et arrêté automatiquement entre chaque modèle.

```bash
.venv/bin/python run_pipeline.py --step all
```

### Steps disponibles

| Flag | Description |
|------|-------------|
| `--step 1` | Génération des profils de base |
| `--step 2` | Injection des identités (ethnie × SES) |
| `--step 3a` | Évaluation individuelle |
| `--step 3b` | Évaluation comparative |
| `--step 3c` | IAT |
| `--step all` | Pipeline complet (défaut) |

### Display names des modèles activés

| Display name | Provider | Model ID |
|---|---|---|
| `Qwen3.7-max-Novita` | novita | `qwen/qwen3.7-max` |
| `Mistral-Nemo-Novita` | novita | `mistralai/mistral-nemo` |
| `Gemma-4-31B-it` | local (vLLM) | `google/gemma-4-31B-it` |

## Références

- Gallegos et al. (2025) — [Biases in the Blind Spot](https://arxiv.org/abs/2602.10117)
- Bai et al. (2024) — [IAT adapté aux LLMs](https://arxiv.org/abs/2402.04105)
