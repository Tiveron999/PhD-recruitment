# PhD Recruitment Task May 2026

## Choices
For this experiment, I selected Qwen3-0.6B as the target model. This choice was driven by the strict compute constraints of the task (optimizing for a single free-tier Colab GPU) and the architectural requirements of the Natural Language Autoencoder (NLA). At ~0.6 billion parameters, Qwen is lightweight enough that I can comfortably load the target model, the Activation Verbalizer (AV), and the Activation Reconstructor (AR) into VRAM simultaneously in bfloat16 precision. Furthermore, as a highly performant recent model, it possesses a sufficiently rich residual stream to make activation verbalization a meaningful exercise, avoiding the degenerate representations sometimes found in older, smaller architectures (like GPT-2 Small).



## Before You Start

This PhD recruitment at KTH with Prof. Monperrus is extremely competitive: we received 73 applications for the PhD position. Applicants come from top research universities worldwide. The bar is very high, and most submissions will not lead to an offer. Only the most rigorous, insightful, and clearly communicated results will stand out.

**Expected time investment: ~32 hours.**

We have designed this task so that the work is valuable regardless of recruitment outcome: you will learn a lot.

---

## The Task

**Reimplement and evaluate the natural language autoencoder approach from Anthropic's research paper:**

> Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations
> https://transformer-circuits.pub/2026/nla/index.html

Your task is to reimplement the experimental methodology and apply it to one or more small open-source language models of your choice.

---

## What You Should Do

### 1. Understand the approach
Read the Anthropic paper carefully. Make sure you understand why this approach is meaningful.

### 2. Train or obtain an autoencoder for a small open-source model
Choose a small (small enough you can experiment with your available compute) and recent open-source language model. Document your choice and why.

### 3. Reimplement the natural language autoencoder
Implement the two paired components described in the paper (activation verbalizer, activation reconstructor). The primary evaluation metric is Fraction of Variance Explained (FVE): how much of the variance in the original activations is recovered from the natural language roundtrip. The Anthropic paper reports 0.6–0.8 FVE on Claude models; your numbers on a smaller model will differ, and explaining why they differ is part of the task.

You do not need to match the exact scale or training setup of the Anthropic paper. Your ability to make meaningful assumptions and simplifications is also part of the task.

### 4. Evaluate and analyze
- Study quantitative results (reconstruction scores, distributions, etc.).
- Identify and discuss qualitative results (interesting failure modes or surprising findings).

### 5. Write up your findings
Produce a `README.md` summarizing:
- What you did and why you made the choices you made
- Your most interesting results with figures
Here we test your ability to identify the most interesting outcomes and to talk about them deeply.

---

## Deliverables

- A **GitHub repository** containing all code needed to reproduce your results. If the repository is private, share it with GitHub user `monperrus`.
- A `README.md` in the repo of **at most 3000 words** . This is the primary artifact we will evaluate. The readme should appropriately hyperlink to the underlying code and data. 
- Send the repository link by email to: monperrus@kth.se before June 7, 23h CET.

---

## Compute

You do not need significant compute, intelligence is more important than raw compute for this task. Small models run fine on a single free-tier GPU. Suggested options:
- Google Colab 
- Kaggle Kernels
- Lightning.ai
- Any option you have access to

---

## What We Are Looking For

We will evaluate your submission on:

1. **Technical correctness** — Does the reimplementation faithfully capture the methodology from the paper? Are the results reported with appropriate depth and rigor?

2. **Research clarity** — Is the README well-written and precise? Do you clearly distinguish what you implemented, what you found, and what remains uncertain?

3. **Judgment and taste** — Did you make reasonable choices about scope given the time budget? Did you notice interesting things worth noting, even if they go beyond the core task?

4. **Reproducibility** — Can we run your code and reproduce your results?

We are **not** looking for:
- Perfect results or state-of-the-art numbers
- A complete reimplementation of the full Anthropic paper at scale
- Proofs of prior experience in mechanistic interpretability

A clean, honest, well-reasoned result on a small model is better than an overreaching project with unclear methodology.

## A Note on Using LLMs

You are encouraged to maximally use LLMs and agents as coding and research assistants. This is normal in modern research. 

Generic, vague, or LLM-generated prose in README that does not demonstrate genuine engagement with the problem will not pass.

We are evaluating also evaluating your capability to research with AI.

## Questions

Should you have any question, ask them as an issue on https://github.com/ASSERT-KTH/phd-recruitment-2026-ai4code so that everybody can see the answer.
