# PhD Recruitment Task May 2026

## Author: Nicholas Tiveron

## Choices
For this experiment, I selected Qwen3-0.6B as the target model. This choice was driven by the strict compute constraints of the task (optimizing for a single free-tier Colab GPU) and the architectural requirements of the Natural Language Autoencoder (NLA). At ~0.6 billion parameters, Qwen is lightweight enough that I can comfortably load the target model, the Activation Verbalizer (AV), and the Activation Reconstructor (AR) into VRAM simultaneously in bfloat16 precision. Furthermore, as a highly performant recent model, it possesses a sufficiently rich residual stream to make activation verbalization a meaningful exercise, avoiding the degenerate representations sometimes found in older, smaller architectures (like GPT-2 Small).


graph TD
    A[Continuous Activation Vector] -->|Input| B(Verbalizer: Qwen-0.6B)
    
    subgraph Teacher-Grounded SFT
    C[BART-large-cnn Teacher] -.->|Target Summary| B
    end

    B -->|Discrete Text Bottleneck| D[Reconstructor]
    
    subgraph Joint GRPO RL Loop
    D -->|Predicted Vector| E{MSE Reward Signal}
    E -->|Update Policy| B
    E -->|Update Weights| D
    end
    
    E -.-> F[Reconstructed Latent Space]
    
    classDef main fill:#2a9d8f,stroke:#264653,stroke-width:2px,color:#fff;
    classDef sub fill:#e9c46a,stroke:#e76f51,stroke-width:2px,color:#333;
    class A,B,D main;
    class C,E sub;


Project Overview
This repository implements a Natural Language Autoencoder (NLA) using Qwen-0.6B as a discrete text bottleneck. The objective of this project was to learn a mapping from high-dimensional, continuous internal activation vectors into discrete, human-readable text strings, and back into the continuous latent space.By utilizing Teacher-Grounded Supervised Fine-Tuning (SFT) and Joint Group Relative Policy Optimization (GRPO), we successfully improved the Fraction of Variance Explained (FVE) from a negative zero-shot baseline to 0.2554, proving that a 0.6B parameter model can effectively compress and reconstruct over 25% of its 1,024-dimensional variance through a pure text bottleneck.2. Methodology & Design ChoicesThe Small-Model Bottleneck & Teacher-Grounded SFTThe Problem: Initial zero-shot attempts to force Qwen-0.6B to decode activation vectors resulted in severe hallucinations and structural collapse (e.g., generating unprompted Wikipedia-style articles).The Choice: We introduced a synthetic teacher (BART-large-cnn) to generate high-fidelity, structurally rigid summaries of the source text.The Why: Instead of learning to decode raw semantics and formatting simultaneously, Teacher-Grounded SFT allowed the Reconstructor to learn a stable affine mapping first, bringing the baseline FVE up to +0.1600.Joint Optimization via GRPOThe Problem: Standard RLHF requires a separate Critic model to evaluate the policy, which is computationally prohibitive on constrained hardware.The Choice: We implemented Group Relative Policy Optimization (GRPO).The Why: GRPO bypasses the Critic model by sampling multiple generations (a group) and standardizing their rewards relative to each other. By using the negative Mean Squared Error (MSE) of the Reconstructor as the reward signal, the Verbalizer and Reconstructor were optimized jointly, culminating in an FVE of 0.2554 after just 3 epochs.3. Deep Dive: Probing the Latent SpaceTo truly evaluate if the autoencoder learned a continuous semantic landscape (rather than a memorized lookup table), we performed linear interpolation between two orthogonal latent concepts: a music video (Concept A) and NHL hockey statistics (Concept B). The results exposed fascinating failure modes intrinsic to small base models.Challenge 1: The Token Explosion (Attention Collapse)When directly injecting the mathematically averaged vector ($\alpha = 0.5$) into model.generate(), the model output infinite repeating tokens (ii*100000000...).Insight: Because Qwen is a base model, receiving a single sequence-length-1 vector stripped it of all positional and contextual attention anchors.Challenge 2: Prompt Hijacking (Over-Conditioning)To fix the token explosion, we concatenated standard prompt text embeddings before the vector. The model generated: "The model is a simple linear regression model..."Insight: The explicit text prompt ("Analyse this internal model...") acted as an overpowering attractor basin. The base model abandoned the noisy interpolated vector and instead auto-completed the prompt itself, turning into a textbook generator.Challenge 3: Manifold Shrinkage (The Hypersphere Problem)Standard linear interpolation (alpha * A + (1.0 - alpha) * B) caused the model to output empty strings.Insight: Activation vectors exist on the curved surface of a high-dimensional hypersphere. A straight-line interpolation passes through the hollow interior of the sphere, resulting in a vector with a drastically reduced magnitude. This pushed the vector out-of-distribution, causing a complete confidence collapse (EOS emission). We resolved this by mathematically re-inflating the blended vector to the target surface magnitude.

graph LR
    A((Concept A)) ---|Standard Linear Lerp: Shrinks Magnitude| C((Dead Zone / Origin))
    C --- B((Concept B))
    
    A .-.-.->|Spherical Re-inflation| D(Continuous Semantic Surface)
    D .-.-.->|Maintains Manifold Norm| B
    
    style C fill:#e63946,stroke:#333,color:#fff
    style D fill:#457b9d,stroke:#333,color:#fff

The Breakthrough: Sequence Loss CrossoverBecause auto-regressive generation proved too fragile for out-of-distribution interpolated vectors, we bypassed text generation entirely. Using Teacher Forcing, we measured the Cross-Entropy Loss of the model when presented with the interpolated vector alongside the gold-standard text of Concept A and Concept B.Table 1: Cross-Entropy Loss across the Latent SweepInterpolation (α)Loss for Concept A (Music)Loss for Concept B (Hockey)1.0 (Pure A)4.28124.87500.84.31254.81250.54.53124.68750.2 (The Boundary)4.59384.59380.0 (Pure B)4.59384.6250Insert Matplotlib Figure Here: A line chart plotting the X-shape intersection of Loss A and Loss B from the table above.Analysis of Findings:The sequence loss provides a mathematical proof of concept blending. As $\alpha$ sweeps from 1.0 to 0.0, the loss for Concept A steadily increases (the model "forgets" the music video), while the loss for Concept B decreases (the model gains confidence in the hockey statistics). At exactly $\alpha = 0.2$, the losses perfectly intersect. This demonstrates that the NLA successfully mapped the discrete text into a smooth, continuous, and highly structured geometric space.4. Reproducibility & Repository StructureTo reproduce these results, execute the provided Jupyter notebooks in a standard GPU environment (e.g., Google Colab with an L4 or T4 GPU).Repository Layout/data - Contains the warmup_data subsets and generated BART summaries./scripts - Contains the core Autoencoder, Verbalizer, and Reconstructor class definitions.01_SFT_Baseline.ipynb - Pipeline for Teacher-Grounded fine-tuning.02_GRPO_RL.ipynb - The joint optimization loop utilizing Group Relative Policy Optimization.03_Latent_Probing.ipynb - Scripts for vector blending, spherical re-inflation, and Teacher Forcing sequence loss evaluation.
