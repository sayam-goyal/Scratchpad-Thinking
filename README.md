# Scratchpad Thinking: Alternation Between Storage and Computation in Latent Reasoning Models

This repository contains the code and data for the paper "Scratchpad Thinking: Alternation Between Storage and Computation in Latent Reasoning Models".

## Abstract

Latent reasoning language models aim to improve reasoning efficiency by computing in continuous hidden space rather than explicit text, but the opacity of these internal processes poses major challenges for interpretability and trust. We present a mechanistic case study of CODI (Continuous Chain-of-Thought via Self-Distillation), a latent reasoning model that solves problems by chaining "latent thoughts". We uncover a structured "scratchpad computation" cycle: even numbered steps serve as scratchpads for storing numerical information, while odd numbered steps perform the corresponding operations. Our results provide a mechanistic account of latent reasoning as an alternating algorithm, demonstrating that non-linguistic thought in LLMs can follow systematic, interpretable patterns. By revealing structure in an otherwise opaque process, this work lays the groundwork for auditing latent reasoning models and integrating them more safely into critical applications.

## The "Scratchpad-Computation" Cycle

Our central finding is that the model executes an alternating "scratchpad-computation" cycle.

* **Even-numbered steps (Scratchpads):** These steps function as scratchpads that store and access numerical information. The model's focus on numeric tokens peaks at these even steps.
* **Odd-numbered steps (Computation):** These steps perform the actual operations. Early-answer forcing shows the largest accuracy gains immediately after odd steps, indicating that key computational updates occur there.

This discovery offers an initial blueprint for the internal algorithms of latent reasoning models.

![Diagram of the Scratchpad-Computation Cycle](https://github.com/sayam-goyal/Scratchpad-Thinking/blob/main/concept_figure.png?raw=true)
*A simplified representation of CODI's reasoning process, illustrating our central finding. The model alternates between Computation Steps (solid outline), where operations are performed, and Scratchpad Steps (dashed outline), which store and represent key numerical information.*

## Methodology

To produce a detailed mechanistic account of CODI's internal algorithm, we used several techniques:

* **SAE-based analysis and steering:** To isolate and identify how specific features are represented and used within CODI's latent space.
* **Activation Patching:** To swap the model's residual stream at every layer for all the latent thoughts to identify components of the model responsible for specific tasks.
* **Early Answer Generation:** To determine how early the model is able to arrive at the correct answer.
* **Attention Analysis:** To understand a model's decision-making process by examining which parts of the input sequence it focuses on when performing a specific computation.
* **Latent Step Manipulation:** To causally probe the model's internal algorithm by actively altering its hidden reasoning steps.

## Conclusion and Limitations

This paper provides mechanistic evidence for a structured, algorithmic process within CODI's latent steps. We show a "scratchpad-computation" cycle where even steps store numeric information and odd steps operate on it. Both observational probes and causal interventions support this alternation, indicating that continuous latent reasoning can be reverse-engineered with mechanistic tools.

Our experiments target one model (GPT-2 CODI) and one domain (GSM8K arithmetic). The next frontier is to test whether this alternating cycle holds in larger models and beyond arithmetic. This work provides the blueprint for those investigations.
