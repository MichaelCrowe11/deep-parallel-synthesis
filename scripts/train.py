#!/usr/bin/env python3

import argparse
import yaml
import json
import torch
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

from dps.model import DPSConfig
from dps.training import DPSTrainer, ScientificReasoningDataset


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Deep Parallel Synthesis Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training data (overrides config)"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        help="Path to evaluation data (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        dest="use_wandb",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.train_data:
        config["data"]["train_data_path"] = args.train_data
    if args.eval_data:
        config["data"]["eval_data_path"] = args.eval_data
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.use_wandb is not None:
        config["training"]["use_wandb"] = args.use_wandb
    
    print("=== Deep Parallel Synthesis Training ===")
    print(f"Configuration loaded from: {args.config}")
    print(f"Output directory: {config['training']['output_dir']}")
    print(f"Training epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print()
    
    model_config = DPSConfig(**config["model"])
    
    trainer = DPSTrainer(
        model_config=model_config,
        training_args=config["training"],
        output_dir=config["training"]["output_dir"]
    )
    
    print("Loading datasets...")
    train_dataset = ScientificReasoningDataset(
        data_path=config["data"]["train_data_path"],
        tokenizer=trainer.tokenizer,
        max_length=config["data"]["max_length"],
        validation_required=config["data"]["validation_required"]
    )
    
    eval_dataset = None
    if config["data"].get("eval_data_path"):
        eval_dataset = ScientificReasoningDataset(
            data_path=config["data"]["eval_data_path"],
            tokenizer=trainer.tokenizer,
            max_length=config["data"]["max_length"],
            validation_required=config["data"]["validation_required"]
        )
    
    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")
    print()
    
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=config["training"]["num_epochs"]
    )
    
    print("\nTraining completed!")
    print(f"Model saved to: {config['training']['output_dir']}")


if __name__ == "__main__":
    main()