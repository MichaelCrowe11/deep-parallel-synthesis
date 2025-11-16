#!/bin/bash

# GPU Training Script for Crowe Logic Framework
# Designed for 10-hour training window on GPU pod

set -e

echo "[v0] Starting Crowe Logic GPU Training Pipeline"

# Setup environment
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="crowe-logic-reasoning"

# Step 1: Export training data from database
echo "[v0] Step 1: Exporting training data..."
python scripts/training/data_exporter.py

# Step 2: Train the model
echo "[v0] Step 2: Starting model training..."
python scripts/training/gpu_trainer.py \
  --model "google/gemma-2-9b-it" \
  --data "./training_data/combined_train.jsonl" \
  --output "./models/crowe_logic_v1" \
  --epochs 3 \
  --batch_size 4 \
  --lr 2e-4

# Step 3: Run evaluation
echo "[v0] Step 3: Evaluating trained model..."
python scripts/training/evaluator.py \
  --model "./models/crowe_logic_v1" \
  --output "./evaluation_results.json"

echo "[v0] GPU Training Pipeline Complete!"
echo "[v0] Model saved to: ./models/crowe_logic_v1"
echo "[v0] Evaluation results: ./evaluation_results.json"
