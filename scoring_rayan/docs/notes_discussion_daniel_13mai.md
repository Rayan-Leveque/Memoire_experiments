# Notes discussion Daniel — 13 mai 2026

Commentaires ajoutés dans les `.tex` suite à la discussion du 13 mai.

---

## OCR — `01_ocr.tex`

### Score legacy Tesseract (mis de côté)

Soit $W^{+}$ les mots détectés avec score Tesseract non nul :

$$s_{\text{legacy}}(x, \hat{y}) = \frac{1}{|W^{+}|} \sum_{i \in W^{+}} \frac{c_i}{100} \in [0, 1]$$

La rotation de l'image retient la transcription maximisant ce score.

### Scoring VLM — deux options proposées

Le VLM ne produit pas de scores par token via l'API actuelle.

**Option A — Confiance verbalisée (recommandée)**

Ajout d'une instruction de post-génération : *"Transcris le texte. Termine par : CONFIANCE: \<entier 0–100\>."*

$$s_{\text{VLM}}(x, \hat{y}) = \frac{c_{\text{verb}}}{100} \cdot \delta_{\text{format}} \in [0, 1]$$

où $\delta_{\text{format}} = 1$ si le token CONFIANCE est présent et parseable, $0.5$ sinon.

**Option B — Self-Consistency ($K$ inférences)**

$$s_{\text{SC}}(x) = 1 - \frac{1}{K(K-1)} \sum_{i \neq j} \mathrm{NED}(\hat{y}_i, \hat{y}_j) \in [0, 1]$$

Coût : $K \times$ latence.

---

## Embeddings — `03_embeddings.tex`

- Tester avec des variations de phrases et des synonymes pour produire un score d'embedding.
- Justifier le choix du modèle d'embedding par comparaison.
- L'embedding est un composant intermédiaire : l'utilisateur final ne le voit pas directement → inutile d'exposer le score.
- **Exception : embedding d'image** → exposer le score de match entre la query et l'image.

---

## LLM Qwen — `05_llm_qwen.tex`

### Protocole d'évaluation des logprobs

Objectif : vérifier que le score de confiance (logprobs) est corrélé avec la qualité de la réponse.

- Prendre des articles existants (articles de journaux Emmanuel).
- Poser une vingtaine de questions : faits, personnes, lieux.
- Deux axes de variation :
  - **Information précise** vs **information large**
  - **Question adaptée au texte** vs **question non adaptée**
- Observer le comportement des logprobs quand la réponse est présente / absente / quand la question est mal posée.
- Signal d'alerte : question non adaptée au texte mais logprobs élevés → mauvais signe.

### Résultats empiriques (19 mai 2026)

Protocole exécuté sur l'article Le Monde du 12 mai 2026 (sommet Africa Forward, Nairobi), 19 questions, 6 modèles (Qwen3-4B, Qwen3.5-0.8B/2B/4B, Qwen3.6-27B, Qwen3.6-35B), 2 variants de prompt.

**Les logprobs sont-ils corrélés avec la qualité de la réponse ?**

Non en valeur absolue, mais le gradient entre quadrants est utilisable sous conditions.

**Avec `explicit_abstain`**

Le signal s'effondre. Les questions non-adaptées (B, D) déclenchent "Je ne sais pas" avec une confiance quasi-parfaite (logprob ≈ 0), ce qui écrase toute discrimination. Le logprob ne mesure plus la qualité de la réponse mais la certitude de ne pas répondre.

**Avec `free` (prompt sans instruction d'abstention)**

Le gradient est lisible sur tous les modèles. L'ordre est systématiquement `D < ... < A` : les questions précises+adaptées (A) génèrent toujours les logprobs les plus élevés. Le signal discrimine bien répondable vs non-répondable. Les valeurs absolues ne sont pas comparables entre modèles (mal calibrées post-RLHF).

**Signal d'alerte**

Aucun cas détecté en `explicit_abstain` : tous les modèles abstiennent correctement sur B et D. En `free`, D est toujours le quadrant le plus bas — le signal ne se déclenche pas non plus, mais pour la bonne raison.

**Conclusion opérationnelle**

| Condition | Résultat |
|---|---|
| Logprob corrélé avec qualité ? | Non — une bonne réponse peut avoir un logprob bas (réponse longue, formulation différente du texte source) |
| Logprob utilisable comme signal de triage ? | Oui, en mode `free`, sur les grands modèles (≥ 27B), comme détection de hors-contexte |
| Seuil de taille minimal | Entre 4B et 27B — les petits modèles n'ont pas de gradient fiable |
| Prompt recommandé pour triage | `free` — le prompt `explicit_abstain` tue le signal |

Recommandation : utiliser le logprob comme feature secondaire de triage (signal faible), couplé au groundedness comme signal principal.
