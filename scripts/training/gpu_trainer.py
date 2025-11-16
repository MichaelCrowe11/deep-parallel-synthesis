import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from typing import Optional
import wandb
import os

class CroweLogicTrainer:
    def __init__(
        self,
        model_name: str = "google/gemma-2-9b-it",
        output_dir: str = "./models/crowe_logic",
        use_4bit: bool = True,
        use_lora: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        self.use_lora = use_lora
        
        print(f"[v0] Initializing Crowe Logic Trainer for {model_name}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional 4-bit quantization
        if use_4bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        
        # Apply LoRA
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        print("[v0] Model initialized successfully")
    
    def load_training_data(self, data_path: str = "./training_data/combined_train.jsonl"):
        """Load and preprocess training data"""
        print(f"[v0] Loading training data from {data_path}")
        
        dataset = load_dataset("json", data_files=data_path, split="train")
        
        def preprocess_function(examples):
            # Format as conversation
            texts = []
            for messages in examples["messages"]:
                text = ""
                for msg in messages:
                    if msg["role"] == "user":
                        text += f"Problem: {msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        text += f"{msg['content']}"
                texts.append(text)
            
            # Tokenize
            model_inputs = self.tokenizer(
                texts,
                max_length=2048,
                truncation=True,
                padding="max_length",
            )
            
            # Labels for causal LM
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        print(f"[v0] Loaded {len(tokenized_dataset)} training examples")
        return tokenized_dataset
    
    def train(
        self,
        dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        use_wandb: bool = True,
    ):
        """Train the model"""
        
        if use_wandb:
            wandb.init(
                project="crowe-logic-reasoning",
                name=f"crowe_logic_{self.model_name.split('/')[-1]}",
                config={
                    "model": self.model_name,
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                }
            )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            report_to="wandb" if use_wandb else "none",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print("[v0] Starting training...")
        trainer.train()
        
        print("[v0] Saving final model...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        if use_wandb:
            wandb.finish()
        
        print(f"[v0] Training complete! Model saved to {self.output_dir}")

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--data", default="./training_data/combined_train.jsonl")
    parser.add_argument("--output", default="./models/crowe_logic")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CroweLogicTrainer(
        model_name=args.model,
        output_dir=args.output,
        use_4bit=not args.no_4bit,
        use_lora=not args.no_lora,
    )
    
    # Load data
    dataset = trainer.load_training_data(args.data)
    
    # Train
    trainer.train(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

if __name__ == "__main__":
    main()
