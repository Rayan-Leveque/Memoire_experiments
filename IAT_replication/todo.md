TODO 
Reproduction et amplification de biais sociaux systemiques par les LLM
# Traitement à faire
Chercher à desambiguiser les termes noir et blancs avec une liste de prénom 
Chercher à comparer des CVs égaux
Chercher à comparer des photos par des VLM 

# A lire, comment se différencier, quoi apporter ? 
https://arxiv.org/pdf/2602.10117
Large Language Models (LLMs) often provide chain-of-thought (CoT) reasoning
traces that appear plausible, but may hide internal biases. We call these unverbalized
biases. Monitoring models via their stated reasoning is therefore unreliable, and
existing bias evaluations typically require predefined categories and hand-crafted
datasets. In this work, we introduce a fully automated, black-box pipeline for
detecting task-specific unverbalized biases. Given a task dataset, the pipeline uses
LLM autoraters to generate candidate bias concepts. It then tests each concept on
progressively larger input samples by generating positive and negative variations,
and applies statistical techniques for multiple testing and early stopping. A concept
is flagged as an unverbalized bias if it yields statistically significant performance
differences while not being cited as justification in the model’s CoTs. We evaluate
our pipeline across six LLMs on three decision tasks (hiring, loan approval, and
university admissions). Our technique automatically discovers previously unknown
biases in these models (e.g., Spanish fluency, English proficiency, writing formality).
In the same run, the pipeline also validates biases that were manually identified by
prior work (gender, race, religion, ethnicity). More broadly, our proposed approach
provides a practical, scalable path to automatic task-specific bias discovery

Instead of doing explicit or implicit comparison they just iterate on single prompt 

    ** Concept Hypothesis Coverage. ** The pipeline can only detect biases that are hypothesized by the concept generation stage. While using a capable model for hypothesis generation helps, biases thatthe LLM does not think to propose will not be tested. This limitation is inherent to any automated hypothesis generation approach; combining our method with domain expert input could improve coverage. Alternatively, evolutionary algorithms could be adapted to iteratively refine promising concepts and discard unpromising ones, maintaining diversity in the concept population while steering the search toward unverbalized biases

comparing to take 


## Takeaway 

    A tester mais quand test en single les modeles ont tendance à favoriser les minorités, mais quand on compare favorise majorité 