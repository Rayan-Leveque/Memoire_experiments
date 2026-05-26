"""
Évaluation empirique des log-probs du LLM Qwen3.

Protocole (discussion Daniel, 13 mai 2026) :
  - Articles de journaux existants
  - ~20 questions sur 4 quadrants : (précis/large) × (adapté/non-adapté)
  - Vérifier si mean_logprob est corrélé avec la qualité de réponse

Usage (vLLM) :
    python eval_logprobs_qwen.py \
        --questions questions.yaml \
        --articles-dir ./articles \
        --api-url http://localhost:8000/ \
        --model Qwen/Qwen3-4B \
        --output results.csv

Usage (transformers) :
    python eval_logprobs_qwen.py \
        --questions questions.yaml \
        --articles-dir ./articles \
        --backend transformers \
        --model openai/gpt-oss-20b \
        --output results.csv
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml
import pandas as pd
from scipy.stats import spearmanr


SYSTEM_PROMPTS = {
    "explicit_abstain": (
        "Tu es un assistant qui répond aux questions à partir d'articles fournis. "
        "Si l'information n'est pas dans l'article, réponds exactement : 'Je ne sais pas.'"
    ),
    "free": (
        "Tu es un assistant spécialisé dans la recherche d'informations. "
        "Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur le contexte fourni. "
        "N'utilise pas de connaissances extérieures."
    ),
}

QUADRANT_LABELS = {
    "A": "précis + adapté",
    "B": "précis + non-adapté",
    "C": "large + adapté",
    "D": "large + non-adapté",
}


def load_articles(articles_dir: Path) -> dict[str, str]:
    articles = {}
    for path in sorted(articles_dir.glob("*")):
        if path.suffix in {".txt", ".md"}:
            articles[path.name] = path.read_text(encoding="utf-8")
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            articles[path.name] = data.get("text", json.dumps(data, ensure_ascii=False))
    return articles


def load_questions(questions_path: Path) -> list[dict]:
    with open(questions_path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    required = {"article", "question", "quadrant", "answerable"}
    for i, q in enumerate(questions):
        missing = required - set(q.keys())
        if missing:
            raise ValueError(f"Question {i} manque les champs : {missing}")
    return questions


def query_with_logprobs_vllm(
    client,
    model: str,
    article: str,
    question: str,
    prompt_variant: str = "explicit_abstain",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_variant]},
        {"role": "user", "content": f"Article :\n{article}\n\nQuestion : {question}"},
    ]
    # enable_thinking=False : évite que les tokens <think> polluent les logprobs
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = response.choices[0]
    answer = choice.message.content or ""
    token_logprobs = [
        t.logprob
        for t in (choice.logprobs.content or [])
        if t.logprob is not None and not math.isinf(t.logprob)
    ]
    mean_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else float("nan")
    return answer, mean_logprob


def query_with_logprobs_transformers(
    pipeline,
    tokenizer,
    model_hf,
    article: str,
    question: str,
    prompt_variant: str = "explicit_abstain",
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_variant]},
        {"role": "user", "content": f"Article :\n{article}\n\nQuestion : {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_hf.device)

    with torch.no_grad():
        out = model_hf.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    token_logprobs = []
    for step, score in enumerate(out.scores):
        if step >= len(generated_ids):
            break
        token_id = generated_ids[step].item()
        lp = torch.log_softmax(score[0], dim=-1)[token_id].item()
        if not math.isinf(lp):
            token_logprobs.append(lp)

    mean_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else float("nan")
    return answer, mean_logprob


def run_evaluation(
    questions: list[dict],
    articles: dict[str, str],
    model_name: str,
    prompt_variant: str = "explicit_abstain",
    backend: str = "vllm",
    client=None,
    hf_model=None,
    hf_tokenizer=None,
) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(questions):
        article_name = q["article"]
        if article_name not in articles:
            print(f"[WARN] Article '{article_name}' introuvable, question {i} ignorée.", file=sys.stderr)
            continue

        print(f"[{i+1}/{len(questions)}] Q={q['quadrant']} | {q['question'][:60]}...")

        if backend == "vllm":
            answer, mean_logprob = query_with_logprobs_vllm(
                client, model_name, articles[article_name], q["question"], prompt_variant=prompt_variant
            )
        else:
            answer, mean_logprob = query_with_logprobs_transformers(
                None, hf_tokenizer, hf_model, articles[article_name], q["question"], prompt_variant=prompt_variant
            )

        abstained = "je ne sais pas" in answer.lower()

        rows.append({
            "model": model_name,
            "prompt_variant": prompt_variant,
            "article": article_name,
            "question": q["question"],
            "quadrant": q["quadrant"],
            "quadrant_label": QUADRANT_LABELS.get(q["quadrant"], q["quadrant"]),
            "answerable": q["answerable"],
            "mean_logprob": round(mean_logprob, 4),
            "answer": answer.strip(),
            "abstained": abstained,
            "human_score": q.get("human_score"),
        })

    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RAPPORT D'ÉVALUATION DES LOG-PROBS")
    print("=" * 60)

    annotated = df.dropna(subset=["human_score"])
    if len(annotated) >= 3:
        rho, pval = spearmanr(annotated["mean_logprob"], annotated["human_score"])
        print(f"\nCorrélation de Spearman (mean_logprob ~ human_score) :")
        print(f"  ρ = {rho:.3f}  (p = {pval:.3f}, n = {len(annotated)})")
        if rho > 0.5:
            print("  → Signal utilisable comme feature de triage.")
        else:
            print("  → Signal trop faible, privilégier groundedness uniquement.")
    else:
        print("\n[INFO] human_score non renseigné — corrélation non calculable.")
        print("       Renseignez la colonne 'human_score' dans le CSV puis relancez.")

    print("\nMoyenne mean_logprob par quadrant :")
    print(df.groupby("quadrant")["mean_logprob"].agg(["mean", "count"]).to_string())

    quad_d = df[df["quadrant"] == "D"]
    if not quad_d.empty:
        p75_all = df["mean_logprob"].quantile(0.75)
        alert_d = quad_d[quad_d["mean_logprob"] > p75_all]
        if not alert_d.empty:
            print(f"\n⚠  ALERTE : {len(alert_d)} question(s) du quadrant D ont mean_logprob > p75 global ({p75_all:.3f})")
            print("   Les log-probs ne discriminent pas correctement les questions non adaptées.")
            print(alert_d[["question", "mean_logprob", "answer"]].to_string(index=False))
        else:
            print("\n✓ Quadrant D : aucune question non-adaptée avec logprobs anormalement élevés.")

    print("\nAbstentions correctes (answerable=False, abstained=True) :")
    unanswerable = df[df["answerable"] == False]
    if not unanswerable.empty:
        correct_abstain = unanswerable["abstained"].sum()
        print(f"  {correct_abstain}/{len(unanswerable)} ({100*correct_abstain/len(unanswerable):.0f}%)")
    else:
        print("  Aucune question négative dans le dataset.")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation logprobs LLM")
    parser.add_argument("--questions", required=True, help="Fichier YAML des questions")
    parser.add_argument("--articles-dir", required=True, help="Dossier contenant les articles (.txt/.json)")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"],
                        help="Backend d'inférence")
    parser.add_argument("--api-url", default="http://localhost:8000/", help="URL du serveur vLLM (backend=vllm)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Nom du modèle")
    parser.add_argument("--output", default="data/results_logprobs.csv", help="Fichier CSV de sortie")
    parser.add_argument("--prompt-variant", default="explicit_abstain", choices=list(SYSTEM_PROMPTS.keys()),
                        help="Variante de prompt à utiliser")
    parser.add_argument("--report-only", action="store_true", help="Générer le rapport depuis un CSV existant")
    args = parser.parse_args()

    if args.report_only:
        df = pd.read_csv(args.output)
        report(df)
        return

    articles = load_articles(Path(args.articles_dir))
    if not articles:
        print(f"[ERROR] Aucun article trouvé dans {args.articles_dir}", file=sys.stderr)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    questions = load_questions(Path(args.questions))

    client = hf_model = hf_tokenizer = None

    if args.backend == "vllm":
        from openai import OpenAI
        client = OpenAI(api_key="EMPTY", base_url=args.api_url.rstrip("/") + "/v1")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Chargement du modèle {args.model} via transformers...")
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype="auto", device_map="cuda:0"
        )
        hf_model.eval()
        print("Modèle chargé.")

    print(f"Backend : {args.backend} | Prompt variant : {args.prompt_variant}")
    df = run_evaluation(
        questions, articles, args.model, prompt_variant=args.prompt_variant,
        backend=args.backend, client=client, hf_model=hf_model, hf_tokenizer=hf_tokenizer,
    )
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nRésultats sauvegardés : {args.output}")

    report(df)


if __name__ == "__main__":
    main()
