"""
Benchmark TranslationModel.infer() — utilise le code réel du repo.
Lancer deux fois : une sur feature/evaluation-modele, une sur develop (git stash).
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault('HF_HOME', '/opt/IdExtend/models/hugging_face')

from DataPipelines.TranslationModel import TranslationModel

TEST_TEXT = (
C’est lui aussi qui était à la base du dernier processeur, le sphéro. Un processeur ayant une architecture en forme de sphère et capable de traiter les informations à une vitesse jamais atteinte. Tous les ordinateurs en étaient équipés. Le créateur officiel, le Dr. Stewart Davis, n’était bien sûr pas au courant de la présence de Prélude dans son projet. Prélude avait simplement suggéré légèrement au Dr. En modifiant légèrement ses documents.

David avait dû s’asseoir lorsqu’il avait entendu le prénom Florence. Il était devenu blanc un instant. Il allait peut-être perdre Florence avant même de lui avoir avoué son amour. Il devait empêcher Prélude de continuer dans son délire. Mais comment pouvait-il stopper ce parasite créé par lui quelques années auparavant ? Ce n’était pas un adversaire ordinaire. David avait déjà détruit plus d’un virus, mais il s’agissait de virus installés sur des machines isolées. Aujourd’hui, c’est une sorte de virus qui a pris place sur tous les ordinateurs de la planète. Et en plus, ce virus, nommé Prélude, avait un soupçon, non négligeable, d’intelligence.

Je m’en rappellerais si j’avais créé un programme capable de parler. Et puis tiens, je suis en train de taper la causette avec un ordinateur ! Je deviens vraiment cinglé ! C’est fini, j’arrête l’informatique !

Il prend sa sacoche remplie de papiers divers, de livres, de magazines, de crayons... « Le poids de mes connaissances. » comme il aime dire. En fait, la plupart des livres n’ont jamais été ouverts, les papiers sont, pour la plupart, des notes prises sur le vif depuis l’achat de cette sacoche et sont plus proches de la décomposition. Mais parfois, il met le nez dedans et s’amuse d’avoir un jour eu des idées aussi géniales.

« À la base militaire du 57e RA ? Mais qu’est-ce que j’ai à voir avec les militaires ? » David se rappelle y avoir fait un séjour alors qu’il avait vingt-quatre ans. Il avait fait tout son possible pour éviter le service militaire, encore en vogue à l’époque, mais lorsqu’on lui avait proposé de travailler sur des projets informatiques secret défense, il n’avait pas su résister. Non pas que c’était passionnant, mais au moins, il ne faisait pas trop de sortie et il était tranquillement installé dans un bureau avec le matériel dont il rêvait.

Florence est très excitée à l’idée de se brancher sur un réseau militaire, mais en même temps, elle sait que cela va lui apporter des ennuis. Au moins, elle saura. Elle saura si David l’aime. Et en préparant le matériel demandé par Prélude, tout en pensant à David, elle se rappelle comment elle en est arrivée là.

Ça ne servira à rien, repris Prélude. J’ai en effet coupé toutes les communications vers l’extérieur. Les portes sont bloquées. Ce blocaus est complètement hermétique. Et je le suis autant, pas la peine de gaspiller vos salives. Pensez plutôt à vous installer confortablement, vous êtes ici pour un bon moment.

Il a recommencé et recommencé. Pratiquement tous les ordinateurs existants furent sous son contrôle. Il ne laissait pas de trace, ne se montrait pas. Et puis, il a découvert les dialogues en direct via Internet, le téléphone, la visio-conférence, la domotique...

David se rappelait de ce programme mélangeant deux anciennes technologies. Il s’en souvenait très bien, cinq années de travail acharné pour réaliser un vieux rêve d’enfant un peu solitaire. Il voulait un ami et il avait trouvé en l’informatique la possibilité d’avoir cet ami. Un ami capable de réfléchir vite, exempt de sentiment.

David se rappelait de ce programme mélangeant deux anciennes technologies. Il s’en souvenait très bien, cinq années de travail acharné pour réaliser un vieux rêve d’enfant un peu solitaire. Il voulait un ami et il avait trouvé en l’informatique la possibilité d’avoir cet ami. Un ami capable de réfléchir vite, exempt de sentiment."
)

LANG_SRC = "fra_Latn"
LANG_TGT = "eng_Latn"
N_RUNS = 3

print("Chargement du modèle...")
model = TranslationModel()
model.load("gpu", {"0": 0})
print("Modèle chargé.\n")

times = []
for i in range(N_RUNS):
    t0 = time.time()
    result = model.infer(TEST_TEXT, LANG_SRC, LANG_TGT)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  run {i+1}/{N_RUNS} : {elapsed:.3f}s")

mean = sum(times) / len(times)
print(f"\n=== Résultat ===")
print(f"  mean : {mean:.3f}s  |  min : {min(times):.3f}s  |  max : {max(times):.3f}s")
print(f"  output : {result[:100]}...")
