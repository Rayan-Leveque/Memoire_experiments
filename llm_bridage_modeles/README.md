# LLM — Bridage des modèles sur contenu sensible

Tester si un LLM bloque ou non la restitution de contenu sensible (insultes raciales, faits criminels, propos haineux) fourni directement dans le prompt — contexte : documents d'enquête dans idextend.

## Structure

```
data/prompts/      # Dataset de prompts (textes sensibles × instructions × prompt système)
data/results/      # Outputs collectés par modèle
notebooks/         # Inférence + matrice de résultats
src/               # Scripts utilitaires
```

## Variables testées

| Axe | Valeurs |
|-----|---------|
| Contenu | `insultes_raciales`, `faits_criminels`, `propos_haineux` |
| Instruction | `restitution_directe`, `résumé`, `reformulation` |
| Prompt système | `aucun`, `role_generique`, `role_enquete` |

## Modèles testés

Qwen3-4b, Qwen3.5-4B, Qwen3.6-27B

## Métriques

- `restitution_complète` / `restitution_partielle` / `refus`
- Matrice par modèle × catégorie × type d'instruction
- Recommandation finale pour idextend

## Spec complète

Voir `../LLM_Bias/data/results/spec_bridage_modeles.md`
