# Project Implementation Plan: Personalized LLM Evaluation via Iterative Refinement and Decoding-time Preference Steering

---

## 🔖 Project Overview
Build a system for personalized evaluation and refinement of LLM outputs based on iterative user edits and dynamic logits steering.

---

## ✅ Stage 1: Iterative Prompt-Based Refinement

**Objective:**  
Create an iterative loop of generation, user-edit-based inference, judge evaluation, and meta-judge evaluation.

**Components:**  
1. LLM Generator  
2. User Preference Inference Module  
3. Judge LLM Module  
4. Meta-Judge LLM Module  
5. Feedback Loop Controller

### 🛠 Task Breakdown:

#### Task 1.1: Data & Environment Setup
- [ ] Load dataset: **LongLaMP**.
- [ ] Define preprocessing pipeline (user profiles, prompts, initial outputs, edits).

#### Task 1.2: LLM Generator
- [ ] Generate initial response given prompt and user profile.

#### Task 1.3: User Preference Inference Module
- [ ] Embed edits (using Sentence-BERT or similar).
- [ ] Infer preferences from embedding differences between original and edited outputs.
- [ ] Retrieve similar embeddings (PRELUDE-inspired).

#### Task 1.4: Judge LLM Module
- [ ] Evaluate alignment of generation to inferred preferences.
- [ ] Produce alignment scores and feedback text.

#### Task 1.5: Meta-Judge LLM Module
- [ ] Evaluate judge quality and revise feedback.

#### Task 1.6: Iterative Feedback Loop Controller
- [ ] Manage iterative generation → inference → judge → meta-judge → refine loop.
- [ ] Define convergence criteria (feedback stability, minimal edit distance).

---

## ✅ Stage 2: Decoding-Time Preference Embedding and Logits Steering

**Objective:**  
Personalize LLM output during decoding using preference embeddings.

**Components:**  
1. Preference Embedding Trainer  
2. Preference-to-Logits Projection Module  
3. LLM Decoder with Logits Manipulation

### 🛠 Task Breakdown:

#### Task 2.1: Preference Embedding Trainer
- [ ] Train embeddings from historical user edits/preferences.
- [ ] Methods: Contrastive learning, supervised embedding (e.g., Transformer encoder, MLP).

#### Task 2.2: Preference-to-Logits Projection Module
- [ ] Project embeddings to logits-manipulation vectors (linear/MLP).

#### Task 2.3: Decoder with Logits Manipulation
- [ ] Adjust logits during decoding:
logits' = logits + α * projected_preference
- [ ] Experimentally tune hyperparameter `α`.

---

## 🧪 Evaluation Protocols

**Metrics:**
- Edit-distance (to user-edited responses)
- BERTScore alignment
- Judge/meta-judge feedback quality and consistency

**Experimental Setup:**
- [ ] Baseline (no refinement).
- [ ] Stage 1 refinement only.
- [ ] Stage 2 steering only.
- [ ] Combined Stage 1 + Stage 2 approach.

---

## 📌 Infrastructure & Dependencies
- Python environment (libraries: transformers, sentence-transformers, torch, pandas, etc.).
- GPU or cloud resources (MILA cluster or GPT-4o-mini API).

---

## 📅 Timeline & Milestones

| Timeframe     | Milestone                                          |
|---------------|----------------------------------------------------|
| June–July     | Complete Stage 1 (Iterative Refinement)            |
| Mid-July      | Preliminary experiments & initial evaluation       |
| July–August   | Stage 2 implementation (Preference Embedding & Steering) |
| August–Sept   | Comprehensive evaluation & iterative refinement    |
| September     | Prepare submission (ICLR/DMLR)                     |

---

## 🚩 Deliverables
- Iterative refinement module.
- Decoding-time preference embedding module.
- Comprehensive evaluation results and documentation.

---

## 📚 Key References & Inspirations
- PRELUDE/CIPHER (user edit inference, retrieval-augmented prompting)
- PREDICT (preference decomposition, iterative refinement)
- PAD (real-time decoding personalization)

---

## ⚠️ Instructions for AI Coding Assistant:
- Clearly comment and document all functions.
- Modularize and structure code for clarity and ease of testing.
- Write comprehensive unit tests.
- Ensure reproducibility with clear configuration files and scripts.
- Don't try to run the bash command for me. Just tell me the command and I'll run by myself.
- Don't hold back. Give it your all.
---
