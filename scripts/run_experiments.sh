#!/bin/bash

# Batch experiment script for running multiple personalization experiments

set -e

echo "Running LLM Personalization Experiments"
echo "========================================"

# Configuration
TASKS=("LongLaMP-2" "LongLaMP-3" "LongLaMP-4")
MODELS=("microsoft/DialoGPT-small" "gpt2" "distilgpt2")
NUM_USERS=10
BETA_VALUES=(0.05 0.1 0.2)

# Create output directory
mkdir -p experiments/batch_run

# Function to run single experiment
run_experiment() {
    local task=$1
    local model=$2
    local beta=$3
    local exp_name="${task}_$(echo $model | tr '/' '_')_beta${beta}"
    
    echo "Running experiment: $exp_name"
    
    python main.py \
        --task "$task" \
        --model_name "$model" \
        --num_users $NUM_USERS \
        --beta $beta \
        --experiment_name "$exp_name" \
        --output_dir "experiments/batch_run" \
        --save_samples \
        --log_level INFO
    
    echo "Completed: $exp_name"
    echo "------------------------"
}

# Run experiments
for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for beta in "${BETA_VALUES[@]}"; do
            run_experiment "$task" "$model" "$beta"
        done
    done
done

echo "All experiments completed!"
echo "Results saved in: experiments/batch_run/"
