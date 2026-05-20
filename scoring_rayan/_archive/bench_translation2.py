import pandas as pd
from DataPipelines.TranslationModel import TranslationModel

# ---------------------------------------------------------------------------
# Test data per source language
# Each entry: (difficulty_label, text)
# "easy"  → well-formed sentences, expected high confidence
# "hard"  → idioms, code-mix, very short, truncated, rare words, etc.
#            → expected to expose low confidence (<0.3/0.4)
# ---------------------------------------------------------------------------
TEST_TEXTS = {
    "fra_Latn": [
        # easy
        ("easy", "C'est lui aussi qui était à la base du dernier processeur, le sphéro."),
        ("easy", "Tous les ordinateurs en étaient équipés."),
        ("easy", "David avait dû s'asseoir lorsqu'il avait entendu le prénom Florence."),
        ("easy", "Il était devenu blanc un instant."),
        ("easy", "Ce n'était pas un adversaire ordinaire."),
        ("easy", "Aujourd'hui, c'est une sorte de virus qui a pris place sur tous les ordinateurs de la planète."),
        ("easy", "Mais parfois, il met le nez dedans et s'amuse d'avoir un jour eu des idées aussi géniales."),
        # hard — syntaxe cassée / mots manquants
        ("hard", "Le chien mange le et puis après rien."),
        ("hard", "Elle parti hier demain viendra pas."),
        ("hard", "Les voitures rouge bleu mange dormir vite."),
        ("hard", "J'ai le truc là le machin tu vois pas le ?"),
        # hard — bruit / fautes massives
        ("hard", "Bnjour comant allez vou aujordhui?"),
        ("hard", "jai manj é du pain avek du beurrre hier soirr"),
        ("hard", "lé zenfan joue dan la jardin"),
        # hard — mélange de langues incohérent
        ("hard", "Je veux aller au supermercado comprar some things für mich."),
        ("hard", "Il a dit que he doesn't want to venir demain con nosotros."),
        ("hard", "Buongiorno, je suis sehr müde aujourd'hui obrigado."),
        # hard — répétitions / boucle
        ("hard", "le le le le le le chat chat chat mange mange mange mange."),
        ("hard", "oui oui oui oui oui oui oui oui oui oui"),
        # hard — symboles / bruit pur
        ("hard", "??? !!! ... ??? !!!"),
        ("hard", "azerty qsdfgh wxcvbn poiuyt"),
        # hard — tronqué au milieu d'un mot
        ("hard", "Il voud"),
        ("hard", "Les enfa"),
    ],
    "eng_Latn": [
        # easy
        ("easy", "The committee approved the new regulations yesterday."),
        ("easy", "She opened the window and looked out at the garden."),
        ("easy", "The report was submitted before the deadline."),
        ("easy", "Children learn languages much faster than adults."),
        ("easy", "The train arrives at platform seven at half past nine."),
        # hard — broken syntax / missing words
        ("hard", "The dog eat the and then nothing after."),
        ("hard", "She go yesterday tomorrow will not came."),
        ("hard", "Cars red blue eats sleeping fast."),
        ("hard", "I have the thing there the thingy you see not the?"),
        # hard — heavy noise / typos
        ("hard", "Helo hw r u todey i am fin thenk yuo"),
        ("hard", "i eted bred wit buttr last nigt"),
        ("hard", "teh childs playz in teh gardn"),
        # hard — incoherent language mix
        ("hard", "I want to aller au supermercado kaufen some Sachen for me."),
        ("hard", "He said que il ne want pas to come mañana avec nous."),
        ("hard", "Buongiorno, I am sehr müde today obrigado."),
        # hard — repetitions / loop
        ("hard", "the the the the cat cat cat eats eats eats eats."),
        ("hard", "yes yes yes yes yes yes yes yes yes yes"),
        # hard — pure noise / symbols
        ("hard", "??? !!! ... ??? !!!"),
        ("hard", "asdfgh qwerty zxcvbn poiuyt"),
        # hard — truncated mid-word
        ("hard", "She wan"),
        ("hard", "The chil"),
    ],
    "por_Latn": [
        # easy
        ("easy", "O relatório foi entregue antes do prazo estipulado."),
        ("easy", "As crianças aprendem idiomas muito mais rápido do que os adultos."),
        ("easy", "O trem chega à plataforma sete às nove e meia."),
        ("easy", "Ela abriu a janela e olhou para o jardim."),
        ("easy", "O comitê aprovou os novos regulamentos ontem."),
        # hard — sintaxe quebrada / palavras faltando
        ("hard", "O cachorro come o e depois nada mais."),
        ("hard", "Ela foi ontem amanhã não virá."),
        ("hard", "Os carros vermelho azul come dormir rápido."),
        ("hard", "Eu tenho a coisa lá o negócio sabe não o?"),
        # hard — ruído / erros massivos
        ("hard", "Oi comu vai vce hoj eu to bien obrigdo"),
        ("hard", "eu comi pao com manteiga onte de note"),
        ("hard", "as criansa brinca no jardin"),
        # hard — mistura de línguas incoerente
        ("hard", "Eu quero go to the supermercado acheter some Sachen para mim."),
        ("hard", "Ele disse que he doesn't want to vir amanhã avec nous."),
        ("hard", "Buongiorno, eu sou sehr müde hoje obrigado."),
        # hard — repetições / loop
        ("hard", "o o o o o gato gato gato come come come come."),
        ("hard", "sim sim sim sim sim sim sim sim sim sim"),
        # hard — símbolos / ruído puro
        ("hard", "??? !!! ... ??? !!!"),
        ("hard", "asdfgh qwerty zxcvbn poiuyt"),
        # hard — truncado no meio de uma palavra
        ("hard", "Ela quer"),
        ("hard", "As crian"),
    ],
}

LIST_TGT = ["eng_Latn", "fra_Latn", "por_Latn", "spa_Latn", "ita_Latn"]

translator = TranslationModel()
translator.load("gpu", {"0": 0})

rows = []
for x, (lang_src, examples) in enumerate(TEST_TEXTS.items()):
    print(f"\n === {x} / {len(TEST_TEXTS)} : {lang_src} ===")
    for j, (difficulty, text) in enumerate(examples):
        for lang_tgt in LIST_TGT:
            if lang_tgt == lang_src:
                continue  # skip identity translation
            result = translator.infer(text, lang_src, lang_tgt)
            confidence = translator.compute_confidence(text, result)
            meta = confidence.metadata

            rows.append({
                "difficulty": difficulty,
                "lang_src": lang_src,
                "lang_tgt": lang_tgt,
                "text": text,
                "text_len": len(text),
                "result": result,
                "confidence_score": confidence.score,
                "mean_beam_prob": meta.get("mean_beam_prob"),
                "min_beam_prob": meta.get("min_beam_prob"),
                "mean_length_ratio": meta.get("mean_length_ratio"),
                "num_chunks": meta.get("num_chunks"),
                "outlier_chunks": meta.get("outlier_chunks"),
            })
            print(f"  [{difficulty}] {lang_src}→{lang_tgt} score={confidence.score:.4f}  | {text} → {result}")

df = pd.DataFrame(rows)
df.to_csv("benchmark_translation.csv", index=True)

print(f"\n=== Résultat par difficulté ===")
print(df.groupby("difficulty")[["confidence_score", "mean_beam_prob", "min_beam_prob"]].mean().round(4).to_string())

print(f"\n=== Résultat par lang_src ===")
print(df.groupby("lang_src")[["confidence_score", "mean_beam_prob"]].mean().round(4).to_string())

print(f"\n=== Résultat par lang_tgt ===")
print(df.groupby("lang_tgt")[["confidence_score", "mean_beam_prob"]].mean().round(4).to_string())

print(f"\n=== Corrélation longueur / score (Pearson) ===")
corr = df[["text_len", "confidence_score"]].corr().iloc[0, 1]
print(f"  r = {corr:.4f}")

print(f"\n=== Score par bucket de longueur (chars) ===")
df["len_bucket"] = pd.cut(df["text_len"], bins=[0, 10, 30, 60, 100, 999], labels=["≤10", "11-30", "31-60", "61-100", ">100"])
print(df.groupby("len_bucket", observed=True)[["confidence_score", "mean_beam_prob"]].mean().round(4).to_string())

print(f"\n=== Cas < 0.4 ===")
low = df[df["confidence_score"] < 0.4][["difficulty", "lang_src", "lang_tgt", "text_len", "confidence_score", "text"]].sort_values("confidence_score")
print(low.to_string())
