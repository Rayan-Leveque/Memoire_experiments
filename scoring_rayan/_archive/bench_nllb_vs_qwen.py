#!/usr/bin/env python3
"""
Benchmark: NLLB-600M vs Qwen3.5 vs TranslateGemma-4B vs Aya-23-8B
Dataset:   FLORES+ devtest
Metrics:   BLEU, chrF++, speed (ms/sentence)

Usage:
    python bench_nllb_vs_qwen.py --models all --n_samples 200
    python bench_nllb_vs_qwen.py --models nllb qwen-0.8b --n_samples 50
    python bench_nllb_vs_qwen.py --models translategemma aya-8b --n_samples 5
    python bench_nllb_vs_qwen.py --models qwen-2b --n_samples 5 --device cpu
"""

import argparse
import csv
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

import sacrebleu
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
)

# ---------------------------------------------------------------------------
# Language pairs to benchmark
# ---------------------------------------------------------------------------
# FLORES+ uses glottocodes for some language variants.
# We use a short alias -> (flores_plus_config, nllb_code) mapping.
LANG_CONFIG = {
    "fra_Latn":     ("fra_Latn",             "fra_Latn"),
    "eng_Latn":     ("eng_Latn",             "eng_Latn"),
    "arb_Arab":     ("arb_Arab",             "arb_Arab"),
    "apc_Arab":     ("apc_Arab_nort3139",    "apc_Arab"),   # North Levantine (Syrian)
    "spa_Latn":     ("spa_Latn",             "spa_Latn"),
    "por_Latn":     ("por_Latn",             "por_Latn"),
    "ita_Latn":     ("ita_Latn",             "ita_Latn"),
}

LANG_PAIRS = [
    ("fra_Latn", "eng_Latn"),
    ("eng_Latn", "fra_Latn"),
    ("fra_Latn", "arb_Arab"),
    ("arb_Arab", "fra_Latn"),
    ("fra_Latn", "apc_Arab"),
    ("apc_Arab", "fra_Latn"),
    ("eng_Latn", "arb_Arab"),
    ("arb_Arab", "eng_Latn"),
]

LANG_NAMES = {
    "fra_Latn": "French",
    "eng_Latn": "English",
    "arb_Arab": "Arabic (MSA)",
    "apc_Arab": "Arabic (North Levantine)",
    "spa_Latn": "Spanish",
    "por_Latn": "Portuguese",
    "ita_Latn": "Italian",
}

# ISO 639-1 codes for TranslateGemma
LANG_ISO = {
    "fra_Latn": "fr",
    "eng_Latn": "en",
    "arb_Arab": "ar",
    "apc_Arab": "ar",  # TranslateGemma has no Levantine variant
    "spa_Latn": "es",
    "por_Latn": "pt",
    "ita_Latn": "it",
}


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------
class ModelWrapper(ABC):
    name: str

    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str: ...


class NLLBWrapper(ModelWrapper):
    name = "nllb-600M"

    def __init__(self, device: str):
        model_id = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        nllb_src = LANG_CONFIG[src_lang][1]
        nllb_tgt = LANG_CONFIG[tgt_lang][1]
        self.tokenizer.src_lang = nllb_src
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        tgt_token_id = self.tokenizer.convert_tokens_to_ids(nllb_tgt)
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_new_tokens=512,
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


class QwenWrapper(ModelWrapper):
    def __init__(self, size: str, device: str, quantize_4bit: bool = False):
        self.name = f"qwen3.5-{size}" + ("-Q4" if quantize_4bit else "")
        model_id = f"Qwen/Qwen3.5-{size}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs = {}
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16 if device != "cpu" else torch.float32
            load_kwargs["device_map"] = device if device != "cpu" else None
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if device == "cpu" and not quantize_4bit:
            self.model = self.model.to("cpu")
        self.model.eval()

    @torch.no_grad()
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_name = LANG_NAMES.get(src_lang, src_lang)
        tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)

        messages = [
            {"role": "system", "content": "You are a professional translator. Output ONLY the translation, nothing else. No explanations, no notes."},
            {"role": "user", "content": f"Translate the following {src_name} text to {tgt_name}:\n\n{text}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        # Slice off the prompt tokens
        output_ids = generated[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Strip any residual thinking tags or markdown
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        response = re.sub(r"^```.*?\n", "", response).rstrip("`").strip()
        # Take only first line if model added explanations
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[0] if lines else response


class TranslateGemmaWrapper(ModelWrapper):
    name = "tgemma-4B"

    def __init__(self, device: str):
        model_id = "google/translategemma-4b-it"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

    @torch.no_grad()
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_iso = LANG_ISO[src_lang]
        tgt_iso = LANG_ISO[tgt_lang]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src_iso,
                        "target_lang_code": tgt_iso,
                        "text": text,
                    }
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        generated = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        output_ids = generated[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(output_ids, skip_special_tokens=True).strip()


class AyaWrapper(ModelWrapper):
    name = "aya-23-8B"

    def __init__(self, device: str):
        model_id = "CohereLabs/aya-23-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

    @torch.no_grad()
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_name = LANG_NAMES.get(src_lang, src_lang)
        tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)
        messages = [
            {"role": "user", "content": f"Translate the following text from {src_name} to {tgt_name}. Output ONLY the translation, nothing else.\n\n{text}"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        generated = self.model.generate(
            input_ids, max_new_tokens=512, do_sample=False,
        )
        output_ids = generated[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[0] if lines else response


# ---------------------------------------------------------------------------
# FLORES-200 loader
# ---------------------------------------------------------------------------
def load_flores_pair(src_lang: str, tgt_lang: str, n_samples: int):
    """Load parallel sentences from FLORES+ devtest (aligned by id)."""
    src_config = LANG_CONFIG[src_lang][0]  # flores_plus config name
    tgt_config = LANG_CONFIG[tgt_lang][0]
    ds_src = load_dataset("openlanguagedata/flores_plus", src_config, split="devtest")
    ds_tgt = load_dataset("openlanguagedata/flores_plus", tgt_config, split="devtest")

    # Build id->text maps and align
    src_map = {row["id"]: row["text"] for row in ds_src}
    tgt_map = {row["id"]: row["text"] for row in ds_tgt}
    common_ids = sorted(set(src_map) & set(tgt_map))[:n_samples]

    sources = [src_map[i] for i in common_ids]
    references = [tgt_map[i] for i in common_ids]
    return sources, references


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return {
        "bleu": round(bleu.score, 2),
        "chrf++": round(chrf.score, 2),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(models: list[ModelWrapper], n_samples: int, output_csv: str):
    all_rows = []
    summary = []

    total_pairs = len(LANG_PAIRS)
    total_models = len(models)

    for m_idx, model in enumerate(models):
        print(f"\n{'='*60}")
        print(f"  Model: {model.name}  ({m_idx+1}/{total_models})")
        print(f"{'='*60}")

        for p_idx, (src, tgt) in enumerate(LANG_PAIRS):
            pair_label = f"{src} -> {tgt}"
            print(f"\n  [{p_idx+1}/{total_pairs}] {pair_label}")

            try:
                sources, references = load_flores_pair(src, tgt, n_samples)
            except Exception as e:
                print(f"    SKIP (dataset load error): {e}")
                continue

            hypotheses = []
            times_ms = []

            for i, src_text in enumerate(sources):
                t0 = time.perf_counter()
                hyp = model.translate(src_text, src, tgt)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                hypotheses.append(hyp)
                times_ms.append(elapsed_ms)

                all_rows.append({
                    "model": model.name,
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "source": src_text,
                    "reference": references[i],
                    "hypothesis": hyp,
                    "time_ms": round(elapsed_ms, 1),
                })

                if (i + 1) % 50 == 0 or i == 0:
                    print(f"    {i+1}/{len(sources)} — {elapsed_ms:.0f}ms — {hyp[:80]}...")

            metrics = compute_metrics(hypotheses, references)
            avg_ms = sum(times_ms) / len(times_ms)

            summary.append({
                "model": model.name,
                "pair": pair_label,
                "n": len(sources),
                "bleu": metrics["bleu"],
                "chrf++": metrics["chrf++"],
                "avg_ms": round(avg_ms, 1),
                "total_s": round(sum(times_ms) / 1000, 1),
            })

            print(f"    => BLEU={metrics['bleu']:.2f}  chrF++={metrics['chrf++']:.2f}  avg={avg_ms:.0f}ms/sent")

    # --- Write detailed CSV (append if exists) ---
    if all_rows:
        file_exists = Path(output_csv).exists()
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDetailed results {'appended to' if file_exists else 'written to'}: {output_csv}")

    # --- Write summary CSV (append if exists) ---
    summary_csv = output_csv.replace(".csv", "_summary.csv")
    if summary:
        file_exists = Path(summary_csv).exists()
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(summary)
        print(f"Summary results {'appended to' if file_exists else 'written to'}:  {summary_csv}")

    # --- Print summary table ---
    print(f"\n{'='*80}")
    print(f"  BENCHMARK SUMMARY — {n_samples} sentences/pair")
    print(f"{'='*80}")
    print(f"{'Model':<16} {'Pair':<30} {'BLEU':>7} {'chrF++':>8} {'ms/sent':>8} {'Total':>7}")
    print("-" * 80)
    for row in summary:
        print(f"{row['model']:<16} {row['pair']:<30} {row['bleu']:>7.2f} {row['chrf++']:>8.2f} {row['avg_ms']:>8.1f} {row['total_s']:>6.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NLLB-600M vs Qwen3.5 on FLORES-200")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["nllb", "qwen-0.8b", "qwen-2b", "qwen-4b", "qwen-9b", "qwen-27b-q4", "translategemma", "aya-8b", "all"],
                        help="Models to benchmark")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of sentences per language pair (max 1012)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda, cpu, or auto")
    parser.add_argument("--output_csv", default="benchmark_nllb_vs_qwen.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    if "all" in args.models:
        model_keys = ["nllb", "qwen-0.8b", "qwen-2b", "qwen-4b", "qwen-9b", "qwen-27b-q4", "translategemma", "aya-8b"]
    else:
        model_keys = args.models

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    models = []
    for key in model_keys:
        print(f"Loading {key}...")
        if key == "nllb":
            models.append(NLLBWrapper(device))
        elif key == "qwen-0.8b":
            models.append(QwenWrapper("0.8B", device))
        elif key == "qwen-2b":
            models.append(QwenWrapper("2B", device))
        elif key == "qwen-4b":
            models.append(QwenWrapper("4B", device))
        elif key == "qwen-9b":
            models.append(QwenWrapper("9B", device))
        elif key == "qwen-27b-q4":
            models.append(QwenWrapper("27B", device, quantize_4bit=True))
        elif key == "translategemma":
            models.append(TranslateGemmaWrapper(device))
        elif key == "aya-8b":
            models.append(AyaWrapper(device))
        print(f"  {key} loaded.")

    run_benchmark(models, args.n_samples, args.output_csv)


if __name__ == "__main__":
    main()
