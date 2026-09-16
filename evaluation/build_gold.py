"""Build and validate the labelled question set in evaluation/gold/.

Each question has:
  doc       which cached document it is about
  kind      lexical (uses the document's wording), paraphrase (avoids it),
            or unanswerable (the document does not contain the answer)
  evidence  short exact strings, any one of which marks a passage as relevant
  pages     1-based PDF pages the evidence is on
  answer    a short reference answer, used to judge correctness
  split     dev or test, fixed by a hash of the id so it never drifts

Validation refuses to write the set if any evidence string is missing from
the extracted text, or appears only on pages other than the ones labelled.
Evidence is kept to 50 characters so it always fits inside one chunk.

    python evaluation/build_gold.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import load_pdf  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
GOLD = HERE / "gold"
MAX_EVIDENCE = 50

DOCUMENTS = {
    "adam": "adam.pdf",
    "attention": "attention.pdf",
    "nist": "nist_ai_rmf.pdf",
    "ipcc": "ipcc_ar6_spm.pdf",
}

L, P, U = "lexical", "paraphrase", "unanswerable"

QUESTIONS: dict[str, list[tuple]] = {
    # (id, kind, question, evidence, pages, answer)
    "adam": [
        ("adam-q01", L, "What are the recommended default hyperparameter values for Adam?",
         ["α = 0.001"], [2],
         "Step size α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10^-8."),
        ("adam-q02", L, "What regret bound does the convergence analysis prove?",
         ["We show Adam has O(√T) regret bound"], [4],
         "An O(√T) regret bound, comparable to the best known bound for general online convex learning."),
        ("adam-q03", L, "How does Adam relate to RMSProp?",
         ["closely related to Adam is RMSProp"], [5],
         "RMSProp is closely related, but RMSProp with momentum applies momentum to the rescaled "
         "gradient while Adam uses running averages of the first and second moments directly, and "
         "RMSProp lacks a bias-correction term."),
        ("adam-q04", L, "Why is the Sum-of-Functions Optimizer impractical on a GPU?",
         ["infeasible on memory-constrained systems"], [5],
         "Its memory requirement is linear in the number of minibatch partitions of the dataset, "
         "which is often infeasible on memory-constrained systems such as a GPU."),
        ("adam-q05", L, "What does bounding the step size by alpha establish around the current parameters?",
         ["establishing a trust region around the current"], [3],
         "A trust region around the current parameter value, beyond which the current gradient "
         "estimate does not provide sufficient information."),
        ("adam-q06", L, "What is AdaMax?",
         ["a variant of Adam based on the infinity norm"], [9],
         "A variant of Adam based on the infinity norm, with its own algorithm (Algorithm 2)."),
        ("adam-q07", L, "Which norm does the AdaMax variant use?",
         ["variant of Adam based on the infinity norm"], [9],
         "The infinity norm."),
        ("adam-q08", L, "Why are the moment estimates biased towards zero initially?",
         ["initialized as (vectors of) 0"], [2],
         "The moving averages are initialized as vectors of zeros, so the estimates are biased "
         "toward zero, especially in the first timesteps and when the decay rates are close to 1."),
        ("adam-q09", L, "Whose online learning framework is the convergence analysis based on?",
         ["online learning framework proposed in (Zinkevich"], [4],
         "Zinkevich's (2003) online learning framework."),
        ("adam-q10", L, "What results are reported on CIFAR-10?",
         ["CIFAR-10 with c64-c64-c128-1000 architecture"], [7],
         "For a c64-c64-c128-1000 CNN on CIFAR-10, Adam and AdaGrad make rapid early progress, but "
         "Adam and SGD eventually converge considerably faster than AdaGrad; Adam shows marginal "
         "improvement over SGD with momentum."),
        ("adam-q11", L, "How was dropout used in the experiments?",
         ["50% dropout noise can be applied to the BoW", "dropout noise is applied to the input layer"],
         [5, 7],
         "50% dropout noise on the IMDB bag-of-words features, dropout-trained multi-layer "
         "networks, and dropout on the input and fully connected layers of the CNN."),
        ("adam-q12", L, "How does Adam compare against SGD with Nesterov momentum?",
         ["Adam yields similar convergence as SGD with", 
          "Adam shows marginal improvement over SGD"], [5, 7],
         "On MNIST logistic regression Adam converges similarly to SGD with Nesterov momentum; on "
         "sparse IMDB features it matches AdaGrad and beats SGD with Nesterov momentum; on the CNN "
         "it shows marginal improvement over SGD with momentum."),
        ("adam-q13", L, "What does the name Adam stand for?",
         ["adaptive moment estimation"], [1],
         "Adaptive moment estimation."),
        ("adam-q14", L, "How are the moment estimates accumulated over time?",
         ["updates exponential moving averages of the"], [2],
         "As exponential moving averages of the gradient and the squared gradient, with decay "
         "rates β1 and β2."),
        ("adam-q15", L, "What architecture was used for the multi-layer neural network experiment?",
         ["two fully connected hidden layers with 1000 hidden"], [6],
         "Two fully connected hidden layers of 1000 ReLU units each, with minibatch size 128."),
        ("adam-q16", L, "How does Adam behave when gradients are sparse?",
         ["In case of sparse gradients, for a reliable"], [3],
         "Sparse gradients need β2 close to 1 for a reliable second-moment estimate; without bias "
         "correction that leads to very large initial steps, so the bias correction is what keeps "
         "Adam stable in this case."),
        ("adam-q17", L, "What is temporal averaging?",
         ["7.2 TEMPORAL AVERAGING"], [9],
         "Averaging the parameters over iterations, for example with an exponential moving average, "
         "because the last iterate is noisy; it often improves generalization."),
        ("adam-q18", L, "Which dataset was used for the logistic regression experiment?",
         ["logistic regression using the MNIST dataset"], [5],
         "MNIST, with IMDB bag-of-words features used for the sparse-feature experiment."),
        ("adam-q19", L, "What convolutional network experiment was run?",
         ["three alternating stages of 5x5 convolution"], [6],
         "A CNN on CIFAR-10 with three stages of 5x5 convolutions and 3x3 max pooling with stride "
         "2, followed by a fully connected layer of 1000 ReLUs."),
        ("adam-q20", L, "How does Adam compare to AdaGrad on sparse features?",
         ["Adam converges as fast as Adagrad"], [6],
         "On sparse IMDB bag-of-words features Adam converges as fast as AdaGrad."),
        ("adam-h01", P, "Who came up with what to call this optimizer?",
         ["coining the name Adam"], [10],
         "Ivo Danihelka and Tom Schaul."),
        ("adam-h02", P, "What does the ratio between the two running estimates tell you about how certain a direction is?",
         ["the signal-to-noise ratio (SNR)"], [3],
         "It is a signal-to-noise ratio: a smaller value means more uncertainty about whether the "
         "direction matches the true gradient, so the effective step shrinks toward zero."),
        ("adam-h03", P, "Is there a limit on how far one update can move the weights?",
         ["establishing a trust region around the current"], [3],
         "Yes. Each step's magnitude is approximately bounded by the step size α."),
        ("adam-h04", P, "Why is that other approach impractical on hardware with limited memory?",
         ["infeasible on memory-constrained systems"], [5],
         "The Sum-of-Functions Optimizer needs memory linear in the number of minibatch partitions, "
         "which is often infeasible on memory-constrained systems such as a GPU."),
        ("adam-h05", P, "Which earlier technique omits the correction and becomes unstable when the decay is near one?",
         ["RMSProp also lacks a bias-correction term"], [5],
         "RMSProp, which lacks bias correction and so takes very large steps and often diverges "
         "when β2 is close to 1."),
        ("adam-h06", P, "What generative model did they train to study the effect of the correction terms?",
         ["(VAE) (Kingma & Welling, 2013)"], [8],
         "A variational autoencoder (VAE)."),
        ("adam-h07", P, "How wide was the hidden layer in that generative experiment?",
         ["single hidden layer with 500 hidden units"], [8],
         "A single hidden layer of 500 softplus units."),
        ("adam-h08", P, "What replaces the squared-gradient accumulator in the alternative version?",
         ["based on the infinity norm"], [9],
         "An infinity-norm accumulator: the exponentially weighted maximum of past gradient magnitudes."),
        ("adam-h09", P, "What starting values do the authors suggest for the tuning knobs?",
         ["α = 0.001"], [2],
         "Step size α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10^-8."),
        ("adam-h10", P, "What is guaranteed about accumulated loss across all rounds?",
         ["We show Adam has O(√T) regret bound"], [4],
         "The regret grows no faster than O(√T)."),
        ("adam-h11", P, "Whose theoretical setting is the proof built on?",
         ["online learning framework proposed in (Zinkevich"], [4],
         "Zinkevich's (2003) online learning framework."),
        ("adam-h12", P, "Which statistical quantity does the second accumulator approximate?",
         ["of the Fisher information matrix (Pascanu"], [5],
         "The diagonal of the Fisher information matrix."),
        ("adam-h13", P, "Which handwritten digit dataset shows up in the experiments?",
         ["logistic regression using the MNIST dataset"], [5],
         "MNIST."),
        ("adam-h14", P, "Was any film review corpus used for testing?",
         ["IMDB movie review dataset"], [5],
         "Yes, IMDB movie reviews, as bag-of-words features over the 10,000 most frequent words."),
        ("adam-h15", P, "What technique did they use to stop the models memorising the training data?",
         ["Stochastic regularization methods, such as dropout"], [6],
         "Dropout (with L2 weight decay also used for the deterministic comparison)."),
        ("adam-h16", P, "Which ten-class image benchmark did they run the convolutional experiment on?",
         ["CIFAR-10 with c64-c64-c128-1000 architecture"], [7],
         "CIFAR-10."),
        ("adam-h17", P, "Which organisation supported this work?",
         ["support of Google Deepmind"], [10],
         "Google DeepMind."),
        ("adam-h18", P, "Does the alternative version recommend a different step size?",
         ["problems are α = 0.002"], [9],
         "Yes. AdaMax uses α = 0.002 instead of Adam's 0.001."),
        ("adam-h19", P, "Which two existing approaches does it claim to bring together?",
         ["combines the advantages of two recently popular"], [10],
         "AdaGrad's ability to handle sparse gradients and RMSProp's ability to handle "
         "non-stationary objectives."),
        ("adam-h20", P, "How does it cope when the thing being optimised keeps changing?",
         ["appropriate for non-stationary objectives"], [1],
         "It is designed to be appropriate for non-stationary objectives."),
        ("adam-h21", P, "Is it affected if each coordinate is scaled differently?",
         ["invariant to diagonal rescaling of the gradients"], [1],
         "No. It is invariant to diagonal rescaling of the gradients."),
        ("adam-h22", P, "Who found the mistake in the original derivation?",
         ["Kai Fan from Duke University"], [10],
         "Kai Fan from Duke University, who spotted an error in the original AdaMax derivation."),
        ("adam-u01", U, "What learning rate did OpenAI use to train GPT-2?", [], [], ""),
        ("adam-u02", U, "How many citations had the Adam paper received by 2020?", [], [], ""),
    ],
    "attention": [
        ("att-01", L, "How many identical layers are in the Transformer's encoder stack?",
         ["The encoder is composed of a stack of N = 6"], [3],
         "Six identical layers."),
        ("att-02", P, "What stops the decoder from looking ahead at words it hasn't produced yet?",
         ["prevent positions from attending to subsequent"], [3],
         "Masking in the decoder self-attention, combined with offsetting the output embeddings by "
         "one position, so predictions depend only on earlier outputs."),
        ("att-03", P, "Why are the dot products divided by the square root of the key size?",
         ["pushing the softmax function into regions where"], [4],
         "For large key dimensions the dot products grow large, pushing the softmax into regions "
         "with extremely small gradients; scaling counteracts this."),
        ("att-04", L, "How many attention heads does the base model use?",
         ["we employ h = 8 parallel attention layers"], [5],
         "Eight heads, each with d_k = d_v = 64."),
        ("att-05", L, "What is the inner-layer dimensionality of the feed-forward network?",
         ["and the inner-layer has dimensionality"], [5],
         "2048, with input and output dimensionality 512."),
        ("att-06", P, "How does the model know the order of the words if it has no recurrence?",
         ["sine and cosine functions of different frequencies"], [6],
         "It adds positional encodings made from sine and cosine functions of different frequencies "
         "to the input embeddings."),
        ("att-07", P, "Did learning the position representations instead of using fixed ones make much difference?",
         ["the two versions produced nearly identical results"], [6],
         "No. Learned positional embeddings gave nearly identical results; sinusoids were kept "
         "because they may extrapolate to longer sequences."),
        ("att-08", L, "What is the maximum path length for a self-attention layer compared with a recurrent layer?",
         ["Self-Attention O(n2 · d) O(1) O(1)"], [6],
         "O(1) for self-attention versus O(n) for a recurrent layer."),
        ("att-09", P, "When is self-attention computationally cheaper than a recurrent layer?",
         ["length n is smaller than the representation"], [7],
         "When the sequence length n is smaller than the representation dimensionality d."),
        ("att-10", L, "How large was the WMT 2014 English-German training set?",
         ["dataset consisting of about 4.5 million"], [7],
         "About 4.5 million sentence pairs."),
        ("att-11", P, "How many tokens of source text went into each training batch?",
         ["approximately 25000 source tokens"], [7],
         "About 25,000 source tokens and 25,000 target tokens."),
        ("att-12", L, "What hardware were the models trained on?",
         ["one machine with 8 NVIDIA P100 GPUs"], [7],
         "One machine with 8 NVIDIA P100 GPUs."),
        ("att-13", P, "How long did it take to train the larger configuration?",
         ["The big models were trained for 300,000 steps"], [7],
         "300,000 steps, about 3.5 days, at 1.0 second per step."),
        ("att-14", L, "How many warmup steps were used in the learning rate schedule?",
         ["warmup_steps = 4000"], [7],
         "4000 warmup steps."),
        ("att-15", P, "What value was used for the Adam beta2 hyperparameter?",
         ["β = 0.9, β = 0.98"], [7],
         "0.98, with β1 = 0.9 and ε = 10^-9."),
        ("att-16", L, "What label smoothing value was used?",
         ["we employed label smoothing of value"], [8],
         "0.1."),
        ("att-17", P, "What downside did label smoothing have?",
         ["hurts perplexity, as the model learns to be more"], [8],
         "It hurts perplexity because the model learns to be more unsure, although it improves "
         "accuracy and BLEU."),
        ("att-18", L, "What BLEU score did the big Transformer reach on English-to-German?",
         ["new state-of-the-art BLEU score of 28.4"], [8],
         "28.4 BLEU."),
        ("att-19", P, "How was the final base model obtained from the saved checkpoints?",
         ["averaging the last 5 checkpoints"], [8],
         "By averaging the last 5 checkpoints, written at 10-minute intervals (the last 20 for big models)."),
        ("att-20", L, "What beam size and length penalty were used for translation?",
         ["beam search with a beam size of 4 and length"], [8],
         "Beam size 4 and length penalty α = 0.6."),
        ("att-21", P, "What happens to quality when using just one attention head?",
         ["single-head attention is 0.9 BLEU worse"], [9],
         "It is 0.9 BLEU worse than the best setting; quality also drops with too many heads."),
        ("att-22", P, "What did shrinking the key dimension suggest about the compatibility function?",
         ["determining compatibility is not easy"], [9],
         "That determining compatibility is not easy and a more sophisticated function than dot "
         "product may be beneficial."),
        ("att-23", L, "What F1 did the 4-layer Transformer get on WSJ section 23 in the semi-supervised setting?",
         ["Transformer (4 layers) semi-supervised 92.7"], [10],
         "92.7 F1."),
        ("att-24", P, "Which earlier parser still beat the Transformer when trained on WSJ alone?",
         ["with the exception of the Recurrent Neural"], [10],
         "The Recurrent Neural Network Grammar."),
        ("att-25", P, "Where can the training and evaluation code be found?",
         ["tensorflow/tensor2tensor"], [10],
         "In the tensor2tensor repository on GitHub (github.com/tensorflow/tensor2tensor)."),
        ("att-26", P, "Who proposed scaled dot-product attention?",
         ["Noam proposed scaled dot-product attention"], [1],
         "Noam Shazeer, who also proposed multi-head attention and the parameter-free position representation."),
        ("att-27", P, "What dropout rate did the big English-to-French model use?",
         ["dropout rate P = 0.1, instead of 0.3"], [8],
         "0.1 instead of 0.3."),
        ("att-28", P, "What value of epsilon was used for label smoothing and dropout in the base model?",
         ["base 6 512 2048 8 64 64 0.1 0.1 100K"], [9],
         "Residual dropout 0.1 and label smoothing ε_ls = 0.1."),
        ("att-u01", U, "How many parameters does GPT-3 have?", [], [], ""),
        ("att-u02", U, "What BLEU score did the Transformer achieve on WMT 2016 Romanian-English?", [], [], ""),
    ],
    "nist": [
        ("nist-01", P, "When is NIST expected to formally review the framework with community input?",
         ["no later than 2028"], [3],
         "No later than 2028."),
        ("nist-02", L, "How does the AI RMF define an AI system?",
         ["an engineered or machine-based system that"], [6],
         "An engineered or machine-based system that, for a given set of objectives, generates "
         "outputs such as predictions, recommendations or decisions influencing real or virtual "
         "environments, with varying levels of autonomy."),
        ("nist-03", P, "Is following this framework mandatory for companies?",
         ["intended to be voluntary, rights-preserving"], [7],
         "No. It is voluntary, rights-preserving, non-sector-specific and use-case agnostic."),
        ("nist-04", L, "Which law directed the development of the AI RMF?",
         ["Artificial Intelligence Initiative Act of 2020"], [7],
         "The National Artificial Intelligence Initiative Act of 2020 (P.L. 116-283)."),
        ("nist-05", L, "What are the four functions of the AI RMF Core?",
         ["GOVERN, MAP, MEASURE, and MANAGE"], [25],
         "GOVERN, MAP, MEASURE and MANAGE."),
        ("nist-06", P, "Which of the four functions is meant to run through all the others?",
         ["GOVERN is a cross-cutting function"], [27],
         "GOVERN, a cross-cutting function infused throughout the other three."),
        ("nist-07", L, "How does the framework define risk?",
         ["the composite measure of an event"], [9],
         "The composite measure of an event's probability of occurring and the magnitude or degree "
         "of its consequences."),
        ("nist-08", P, "Does the framework tell organisations how much risk they should accept?",
         ["it does not prescribe risk tolerance"], [12],
         "No. It can be used to prioritize risk but does not prescribe risk tolerance, which is "
         "highly contextual."),
        ("nist-09", P, "What should happen if an AI system poses unacceptable risk?",
         ["should cease in a safe manner"], [13],
         "Development and deployment should cease in a safe manner until the risks can be "
         "sufficiently managed."),
        ("nist-10", L, "How is residual risk defined?",
         ["risk remaining after risk treatment"], [13],
         "Risk remaining after risk treatment."),
        ("nist-11", L, "What are the characteristics of trustworthy AI?",
         ["valid and reliable, safe, secure and resilient"], [8, 17],
         "Valid and reliable; safe; secure and resilient; accountable and transparent; explainable "
         "and interpretable; privacy-enhanced; and fair with harmful bias managed."),
        ("nist-12", P, "Which trustworthiness characteristic is the foundation for all the others?",
         ["Valid & Reliable is a necessary condition"], [17],
         "Valid and reliable, a necessary condition shown as the base for the other characteristics."),
        ("nist-13", P, "What is the difference between resilience and security?",
         ["Security and resilience are related but distinct"], [20],
         "Resilience is the ability to return to normal function after an unexpected adverse event; "
         "security includes resilience but also covers protocols to avoid, protect against, respond "
         "to or recover from attacks."),
        ("nist-14", L, "Name some common security concerns for AI systems.",
         ["adversarial examples, data poisoning"], [20],
         "Adversarial examples, data poisoning, and exfiltration of models, training data or other "
         "intellectual property through AI system endpoints."),
        ("nist-15", P, "What question does interpretability answer, compared with explainability?",
         ["pretability can answer the question of"], [22],
         "Interpretability answers why a decision was made and what it means to the user; "
         "explainability answers how it was made; transparency answers what happened."),
        ("nist-16", L, "What are the three major categories of AI bias?",
         ["computational and statistical, and human"], [23],
         "Systemic, computational and statistical, and human-cognitive."),
        ("nist-17", P, "Can a privacy protection technique make a model less accurate?",
         ["techniques can result in a loss in accuracy"], [17, 22],
         "Yes. Under conditions such as data sparsity, privacy-enhancing techniques can reduce "
         "accuracy, which affects decisions about fairness."),
        ("nist-18", P, "Is a system with balanced predictions across demographic groups necessarily fair?",
         ["mitigated are not necessarily fair"], [22],
         "No. It may still be inaccessible to people with disabilities or affected by the digital "
         "divide, or exacerbate existing disparities or systemic biases."),
        ("nist-19", L, "After GOVERN, which function do most users start with?",
         ["would start with the MAP function"], [26],
         "MAP, continuing to MEASURE or MANAGE."),
        ("nist-20", P, "What decision should an organisation be able to make after completing MAP?",
         ["initial go/no-go decision"], [30],
         "An initial go/no-go decision about whether to design, develop or deploy the AI system."),
        ("nist-21", L, "What does GOVERN 1.7 cover?",
         ["GOVERN 1.7: Processes and procedures are in place"], [28],
         "Processes for decommissioning and phasing out AI systems safely, without increasing risk "
         "or decreasing the organization's trustworthiness."),
        ("nist-22", P, "Who should take responsibility for decisions about AI risks according to the governance subcategories?",
         ["Executive leadership of the organization takes"], [28],
         "Executive leadership of the organization."),
        ("nist-23", P, "Should the people who build an AI model also be the ones who verify it?",
         ["separated from those verifying and validating"], [5, 16],
         "No. As a best practice, those building and using models are separated from those "
         "verifying and validating them."),
        ("nist-24", L, "What does MEASURE 2.12 assess?",
         ["Environmental impact and sustainability of AI"], [35],
         "The environmental impact and sustainability of AI model training and management activities."),
        ("nist-25", P, "What options can an organisation choose when responding to a high-priority AI risk?",
         ["Risk response options can include mitigating"], [37],
         "Mitigating, transferring, avoiding or accepting the risk."),
        ("nist-26", L, "What is the difference between a Current Profile and a Target Profile?",
         ["A Target Profile indicates the"], [38],
         "A Current Profile shows how AI is currently managed and the related risks; a Target "
         "Profile shows the outcomes needed to reach the desired risk management goals. Comparing "
         "them reveals gaps."),
        ("nist-27", P, "Does the framework provide templates for profiles?",
         ["does not prescribe profile templates"], [39],
         "No. It does not prescribe profile templates, to allow flexibility in implementation."),
        ("nist-28", P, "Who are the key actors responsible for AI governance in an organization?",
         ["senior leadership, and the Board of Directors"], [42],
         "Organizational management, senior leadership and the Board of Directors."),
        ("nist-29", P, "How can the data behind an AI system become a risk over time?",
         ["may become stale or outdated"], [43],
         "Training datasets can become detached from their original context or stale and outdated "
         "relative to the deployment context."),
        ("nist-30", L, "What kind of language does the AI RMF aim to use?",
         ["Use clear and plain language"], [47],
         "Clear and plain language understandable by a broad audience, including people who are "
         "not AI professionals, while technically deep enough for practitioners."),
        ("nist-u01", U, "How much funding did NIST spend developing the AI RMF?", [], [], ""),
        ("nist-u02", U, "Which companies had adopted the AI RMF by 2024?", [], [], ""),
    ],
    "ipcc": [
        ("ipcc-01", L, "How much warmer was global surface temperature in 2011-2020 than in 1850-1900?",
         ["1.09 [0.95 to 1.20]°C"], [10],
         "1.09°C (0.95 to 1.20°C)."),
        ("ipcc-02", P, "Did land or ocean warm more?",
         ["larger increases over land"], [10],
         "Land warmed more: 1.59°C over land versus 0.88°C over the ocean."),
        ("ipcc-03", L, "What were global net anthropogenic GHG emissions in 2019?",
         ["59 ± 6.6 GtCO"], [10],
         "About 59 ± 6.6 GtCO2-eq, about 12% higher than in 2010 and 54% higher than in 1990."),
        ("ipcc-04", L, "How much did global mean sea level rise between 1901 and 2018?",
         ["0.20 [0.15 to 0.25] m between 1901 and 2018"], [11],
         "0.20 m (0.15 to 0.25 m)."),
        ("ipcc-05", P, "How fast has the sea been rising recently?",
         ["3.7 [3.2 to 4.2] mm yr-1"], [11],
         "3.7 mm per year between 2006 and 2018, up from 1.3 mm per year in 1901-1971."),
        ("ipcc-06", L, "How many people live in contexts highly vulnerable to climate change?",
         ["3.3 to 3.6 billion people"], [11],
         "Approximately 3.3 to 3.6 billion people."),
        ("ipcc-07", P, "How much deadlier were floods, droughts and storms in highly vulnerable regions?",
         ["15 times higher in highly vulnerable regions"], [11],
         "Between 2010 and 2020 mortality was 15 times higher than in regions with very low vulnerability."),
        ("ipcc-08", P, "What share of the world's people face severe water scarcity at some point in the year?",
         ["Roughly half of the world"], [12],
         "Roughly half, for at least part of the year."),
        ("ipcc-09", L, "How many countries include adaptation in their climate policies?",
         ["at least 170 countries"], [14],
         "At least 170 countries, as well as many cities."),
        ("ipcc-10", P, "What are the main obstacles to adaptation?",
         ["Key barriers to adaptation are limited resources"], [15],
         "Limited resources, lack of private sector and citizen engagement, insufficient finance, "
         "low climate literacy, lack of political commitment, limited research or slow uptake of "
         "adaptation science, and a low sense of urgency."),
        ("ipcc-11", P, "By how much did the cost of solar energy fall between 2010 and 2019?",
         ["unit costs of solar energy (85%)"], [17],
         "85%. Wind energy fell 55% and lithium-ion batteries 85%."),
        ("ipcc-12", L, "What warming is projected by 2100 without strengthening policies?",
         ["global warming of 3.2 [2.2 to 3.5] °C"], [17],
         "3.2°C (2.2 to 3.5°C)."),
        ("ipcc-13", P, "What warming results if countries only meet the pledges announced before COP26?",
         ["median global warming of 2.8 [2.1 to 3.4] °C"], [17],
         "A median of 2.8°C (2.1 to 3.4°C) by 2100."),
        ("ipcc-14", L, "What was the climate finance goal from developed to developing countries?",
         ["USD 100 billion per year by 2020"], [17],
         "USD 100 billion per year by 2020; flows in 2018 were below this goal."),
        ("ipcc-15", P, "What is the best estimate of warming by 2081-2100 under a very high emissions scenario?",
         ["4.4°C for a very high GHG emissions scenario"], [18],
         "4.4°C under SSP5-8.5, compared with 1.4°C for very low and 2.7°C for intermediate emissions."),
        ("ipcc-16", P, "How much can a single year's global temperature vary naturally?",
         ["about ±0.25°C"], [18],
         "About ±0.25°C (5-95% range)."),
        ("ipcc-17", L, "How often do large explosive volcanic eruptions occur?",
         ["occur on average twice per century"], [19],
         "On average twice per century; such an eruption would temporarily cool the surface for one "
         "to three years."),
        ("ipcc-18", P, "How much could sea level rise by 2100 under a very high emissions scenario?",
         ["0.63–1.01 m by 2100"], [24],
         "0.63 to 1.01 m relative to 1995-2014 (likely range)."),
        ("ipcc-19", L, "At what warming level will the Greenland and West Antarctic ice sheets be lost almost completely?",
         ["warming levels between 2°C and 3°C"], [24],
         "At sustained warming between 2°C and 3°C, over multiple millennia."),
        ("ipcc-20", P, "How much does temperature rise for every trillion tonnes of CO2 emitted?",
         ["global surface temperature rises by 0.45°C"], [25],
         "0.45°C per 1000 GtCO2 (likely range 0.27 to 0.63°C)."),
        ("ipcc-21", L, "What is the remaining carbon budget for a 50% chance of limiting warming to 1.5°C?",
         ["500 GtCO for a 50% likelihood"], [25],
         "500 GtCO2 from the beginning of 2020."),
        ("ipcc-22", P, "By when do pathways that hold warming to 1.5°C reach net zero CO2?",
         ["in the early 2050s and around the early"], [26],
         "In the early 2050s; pathways limiting warming to 2°C reach it around the early 2070s."),
        ("ipcc-23", L, "By what percentage must GHG emissions fall by 2030 in 1.5°C pathways?",
         ["GHG 43 [34-60]"], [27],
         "43% (34 to 60%) below 2019 levels."),
        ("ipcc-24", P, "By how much must methane fall by 2030 to stay on a 1.5°C track?",
         ["reduced by 34 [21–57] %"], [27],
         "34% (21 to 57%) relative to 2019."),
        ("ipcc-25", P, "How much CO2 could be stored underground?",
         ["on the order of 1000 GtCO"], [27],
         "On the order of 1000 GtCO2, more than needed through 2100 to limit warming to 1.5°C."),
        ("ipcc-26", P, "Which mitigation options cost less than USD 20 per tonne?",
         ["costs less than USD 20 tCO"], [34],
         "Solar and wind energy, energy efficiency improvements, and methane emissions reductions "
         "from coal mining, oil and gas, and waste."),
        ("ipcc-27", L, "What share of Earth's land, freshwater and ocean should be conserved to keep biodiversity resilient?",
         ["approximately 30% to 50% of Earth"], [35],
         "Approximately 30% to 50%."),
        ("ipcc-28", P, "How much would removing fossil fuel subsidies cut CO2 emissions?",
         ["reduce global CO emission by 1 to 4%"], [38],
         "By 1 to 4% of global CO2 emissions, and GHG emissions by up to 10%, by 2030."),
        ("ipcc-29", P, "How much does mitigation investment need to grow this decade?",
         ["three to six greater than current levels"], [39],
         "Average annual modelled mitigation investment needs for 2020-2030 are three to six times "
         "current levels."),
        ("ipcc-30", L, "How is the near term defined in the report?",
         ["near term is defined as the period until 2040"], [9],
         "The period until 2040; the long term is beyond 2040."),
        ("ipcc-u01", U, "What will global temperature be in 2150 under the intermediate emissions scenario?", [], [], ""),
        ("ipcc-u02", U, "What was the total budget for producing the Sixth Assessment Report?", [], [], ""),
    ],
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def split_for(question_id: str) -> str:
    return "dev" if int(hashlib.sha1(question_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def main() -> int:
    GOLD.mkdir(exist_ok=True)
    problems = []
    for doc, items in QUESTIONS.items():
        pages = {p.number: normalise(p.text) for p in load_pdf((CACHE / DOCUMENTS[doc]).read_bytes())}
        rows = []
        for qid, kind, question, evidence, gold_pages, answer in items:
            for ev in evidence:
                if len(ev) > MAX_EVIDENCE:
                    problems.append(f"{qid}: evidence longer than {MAX_EVIDENCE}: {ev!r}")
            if kind != U:
                found = {n for n, text in pages.items() for ev in evidence if normalise(ev) in text}
                if not found:
                    problems.append(f"{qid}: no evidence string found in the document")
                elif not found & set(gold_pages):
                    problems.append(f"{qid}: evidence found on {sorted(found)}, labelled {gold_pages}")
                gold_pages = sorted(found | set(gold_pages) & found)
            rows.append({
                "id": qid, "doc": doc, "kind": kind, "question": question,
                "evidence": evidence, "pages": gold_pages, "answer": answer,
                "split": split_for(qid),
            })
        (GOLD / f"{doc}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
        dev = sum(r["split"] == "dev" for r in rows)
        print(f"{doc}: {len(rows)} questions ({dev} dev, {len(rows) - dev} test)")
    if problems:
        print("\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
