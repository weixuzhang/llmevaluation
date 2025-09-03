Personalized Generation with User Embeddings 

Project Overview
We are developing a prototype system for personalized text generation using large language models (LLMs). The goal is to demonstrate that incorporating a simple user embedding—extracted from each user’s profile or historical writing—can improve the personalization of generated outputs.

Experiment Objective
Use the LongLaMP benchmark (personalized long-form text generation) as the testbed.

For each user, extract a basic user embedding vector from their historical text data.

During inference, steer the LLM’s generation by adjusting the logits for each candidate token based on its similarity to the user embedding.

Compare the personalized model’s outputs to a baseline (LLM without personalization) using ROUGE and METEOR metrics.

High-Level Steps
Data Loading: Load the LongLaMP dataset and select a subset of users for prototyping.

User Embedding Construction:

For each user, process their profile/history to generate a simple embedding (e.g., average Sentence-BERT vector).

Personalized Generation:

At each decoding step, compute the similarity between candidate token representation and the user embedding.

Adjust LLM logits (e.g., by adding $\beta \times$ similarity score to the logit for each candidate token).

Generate the text sequence using the modified logits.

Evaluation:

Generate texts with and without personalization for each user/input.

Compute ROUGE and METEOR scores for both settings.

Report improvements and sample outputs.

Deliverables
Python code with clear structure (data preprocessing, embedding extraction, generation loop, evaluation).

Documentation/comments explaining each step.

Output: quantitative results (ROUGE/METEOR) and several example generations.

Notes
Start with a small number of users for the initial prototype.

Use open-source LLMs and Sentence-BERT if possible.

The focus is on proof-of-concept, not exhaustive model tuning.

