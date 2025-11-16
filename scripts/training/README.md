# Crowe Logic GPU Training Pipeline

Complete pipeline for training reasoning models on GPU pods.

## Quick Start

### 1. Export Training Data
\`\`\`bash
python scripts/training/data_exporter.py
\`\`\`

This fetches all high-quality reasoning sessions from the database and formats them for training.

### 2. Run GPU Training (10-hour window)
\`\`\`bash
bash scripts/training/run_gpu_training.sh
\`\`\`

The complete pipeline includes:
- Data export from Supabase
- Model fine-tuning with LoRA + 4-bit quantization
- Automatic evaluation on test set
- W&B logging for metrics

### 3. Model Configuration

**Default Setup:**
- Base Model: `google/gemma-2-9b-it`
- LoRA rank: 16
- 4-bit quantization (NF4)
- Batch size: 4 with gradient accumulation
- Learning rate: 2e-4
- Epochs: 3

**Estimated Training Time:**
- ~8 hours on A100 GPU
- ~10 hours on V100 GPU
- ~6 hours on H100 GPU

### 4. Custom Training

\`\`\`bash
python scripts/training/gpu_trainer.py \
  --model "meta-llama/Llama-3-8B" \
  --data "./training_data/math_train.jsonl" \
  --output "./models/custom_model" \
  --epochs 5 \
  --batch_size 8 \
  --lr 1e-4
\`\`\`

## GPU Pod Setup

### Requirements
\`\`\`
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
datasets>=2.18.0
wandb>=0.16.0
accelerate>=0.28.0
\`\`\`

### Environment Variables
\`\`\`bash
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_huggingface_token"
\`\`\`

## Output Structure

\`\`\`
models/
└── crowe_logic_v1/
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── tokenizer_config.json
    └── special_tokens_map.json

evaluation_results.json
training_data/
├── math_train.jsonl
├── logic_train.jsonl
├── science_train.jsonl
├── code_train.jsonl
└── combined_train.jsonl
\`\`\`

## Next Steps

After training, deploy the model to use in the reasoning API by updating the model name in `app/api/reason/route.ts`.
