Preference heads
Beyond Retrieval: Discovering and Steering Preference Heads for User Alignment

1. Project Vision
This project introduces Differential Preference Steering (DPS), a novel, training-free framework for personalizing Large Language Models (LLMs) with fine-grained and interpretable control. We move beyond treating the model as a black box and instead intervene directly on the internal mechanisms responsible for user adaptation.
Our central thesis is grounded in the emerging field of mechanistic interpretability. Recent work has shown that for complex tasks like In-Context Learning (ICL), the key drivers are not simple pattern-matching circuits (like "induction heads"), but more sophisticated Function Vector (FV) Heads that learn an abstract representation of the task at hand (Yin & Steinhardt, 2025).
We posit that a similar, more sophisticated mechanism governs personalization. We introduce the concept of Preference Heads: a specialized set of attention heads that are causally responsible for encoding a user's multi-dimensional profile—including their writing style, topical focus, structural habits, and vocabulary—into a latent "user preference vector." By identifying these heads, we can adapt powerful contrastive decoding techniques to precisely steer the model's generation, treating a generic response as a deviation from the user's learned preference function.

2. The Core Problem & The Gap in Understanding
Current personalization methods are limited by their inability to precisely control the model's internal representations:
●Coarse-grained methods like RAG or single-vector steering lack the precision for nuanced user alignment.
●Fine-grained methods like full fine-tuning are computationally prohibitive and operationally impractical for individual users.
Crucially, prior work on localizing model behavior, such as the discovery of "retrieval heads" (Wu et al., 2024), focused on heads that perform simple copy-paste operations—analogous to the less powerful "induction heads." This is insufficient for the complex, semantic task of personalization. The field lacks a method for identifying and leveraging the more sophisticated, abstracting heads that truly govern user-specific generation. Our work fills this critical gap.

3. Our Proposed Framework: Differential Preference Steering (DPS)
DPS is a decoding-time algorithm that operates in two stages: identifying the core mechanistic components (Preference Heads) and then using them to steer the model.
3.1. Stage 1: Identifying Preference Heads
1. We will develop a robust methodology to identify the attention heads that function as encoders of a user's preference vector. This requires moving beyond simple attention or copy-paste metrics.Task Formulation: Using the LongLaMP benchmark, we will provide the LLM with a user's historical data (their "profile") followed by a new task prompt.
2.Preference Contribution Score: Our metric will be designed to identify heads that are causally responsible for abstracting the user's profile into a functional representation. This score will measure a head's contribution to generating a user-aligned output when its activation is conditioned on the user profile.
3.Validation: We will prove the functional importance of these heads through rigorous ablation studies. We will demonstrate that masking our identified Preference Heads severely degrades personalization, while masking simpler "retrieval" or "induction-like" heads has a significantly smaller impact, aligning with the findings of Yin & Steinhardt (2025).
3.2. Stage 2: Contrastive Decoding with Preference Heads
1. At each decoding step, DPS performs two parallel forward passes to isolate the user preference signal:The Personalized Pass: The full, unmodified model generates logits, guided by the "user preference vector" formed by the Preference Heads. logits_personalized = LLM(context, user_profile)
2.The Generic Pass: A "preference-agnostic" version of the model, created by masking the identified Preference Heads, generates baseline logits. This pass represents the model's output without the ability to form a coherent user preference vector. logits_generic = LLM_masked(context, user_profile)
3.Differential Steering: The final logits are calculated by amplifying the difference between the two distributions, effectively steering the generation towards outputs that are strongly supported by the user's preference function. logit_final ∝ (1 + γ) * logit_personalized - γ * logit_generic

4. Key Innovations and Differentiators
●Discovery and Mechanistic Characterization of Preference Heads: Our primary contribution is the identification and functional characterization of a new class of specialized attention heads, establishing them as the personalization equivalent of Function Vector (FV) heads.
●Mechanistically-Grounded Personalization: We are the first to propose a personalization framework based on intervening on these sophisticated, abstracting components, offering a new level of precision and control.
●Novel Application of Contrastive Decoding: We adapt this powerful technique from the domain of factuality to the more complex and semantic domain of user preference alignment.
●Training-Free and Interpretable: DPS is an inference-time algorithm that provides a clear, inspectable personalization signal (Δ_logits) at every generation step.

5. High-Level Experiment Plan
1.Phase 1: Identify and Validate Preference Heads:Objective: To empirically identify Preference Heads in a model like Llama-3-8B and validate their causal role in personalization.
a.Method: Use our "Preference Contribution Score" on the LongLaMP dataset. Validate via ablation studies, comparing the impact of masking Preference Heads versus masking baseline "retrieval/induction" heads.
2.Phase 2: Validate DPS Framework:Objective: To prove that DPS provides superior personalization over existing methods.
a.Baselines: Compare against RAG, StyleVector, and a control version of DPS that uses the simpler "retrieval heads."
b.Metrics: Measure content quality (ROUGE-L) and preference alignment (using an LLM-as-a-Judge).

6. Expected Impact
This research will make two significant contributions. First, it will deliver a practical, high-performance, and interpretable framework (DPS) for LLM personalization. Second, and more fundamentally, it will advance the field of mechanistic interpretability by identifying and characterizing Preference Heads, providing a deeper understanding of how Large Language Models learn and represent complex, user-specific functions.
