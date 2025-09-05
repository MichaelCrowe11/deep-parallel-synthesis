import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import deepspeed
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from trl import SFTTrainer, DPOTrainer
from datasets import load_dataset, Dataset as HFDataset
import wandb
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .model import DPSModel, DPSConfig
from .reasoning import ParallelReasoningChains
from .validator import ScientificValidator, ValidationStatus


class ScientificReasoningDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        validation_required: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.validation_required = validation_required
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.validator = ScientificValidator() if validation_required else None
        self._preprocess_data()
    
    def _preprocess_data(self):
        processed_data = []
        for item in tqdm(self.data, desc="Preprocessing data"):
            if self.validation_required and self.validator:
                validation_result = self.validator.validate(
                    content=item.get("response", ""),
                    reasoning_type=item.get("reasoning_type", "deductive"),
                    evidence=item.get("evidence", [])
                )
                
                if validation_result.is_valid():
                    item["validation_score"] = validation_result.confidence
                    processed_data.append(item)
            else:
                processed_data.append(item)
        
        self.data = processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item["prompt"]
        response = item["response"]
        
        full_text = f"### Scientific Query:\n{prompt}\n\n### Scientific Response:\n{response}"
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = encoding["input_ids"].clone()
        
        prompt_tokens = self.tokenizer(
            f"### Scientific Query:\n{prompt}\n\n### Scientific Response:\n",
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        
        labels[:, :len(prompt_tokens)] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "validation_score": torch.tensor(item.get("validation_score", 1.0)),
        }


class DPSTrainer:
    def __init__(
        self,
        model_config: DPSConfig,
        training_args: Dict[str, Any],
        output_dir: str = "./dps_output"
    ):
        self.model_config = model_config
        self.training_args = training_args
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = DPSModel(model_config)
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 1),
            mixed_precision=training_args.get("mixed_precision", "bf16"),
            log_with="wandb" if training_args.get("use_wandb", True) else None,
        )
        
        self.deepspeed_config = self._create_deepspeed_config()
        
        if training_args.get("use_wandb", True):
            wandb.init(
                project="deep-parallel-synthesis",
                config={
                    **model_config.to_dict(),
                    **training_args
                }
            )
    
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        return {
            "train_batch_size": self.training_args.get("total_batch_size", 32),
            "gradient_accumulation_steps": self.training_args.get("gradient_accumulation_steps", 4),
            "gradient_clipping": self.training_args.get("gradient_clipping", 1.0),
            "fp16": {
                "enabled": self.training_args.get("mixed_precision") == "fp16",
            },
            "bf16": {
                "enabled": self.training_args.get("mixed_precision") == "bf16",
            },
            "zero_optimization": {
                "stage": self.training_args.get("zero_stage", 3),
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": self.training_args.get("num_train_steps", 10000),
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.training_args.get("learning_rate", 5e-5),
                    "warmup_num_steps": self.training_args.get("warmup_steps", 1000)
                }
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.training_args.get("learning_rate", 5e-5),
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": self.training_args.get("weight_decay", 0.01)
                }
            },
            "tensorboard": {
                "enabled": True,
                "output_path": str(self.output_dir / "tensorboard"),
                "job_name": "dps_training"
            }
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3
    ):
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_args.get("batch_size", 4),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.training_args.get("eval_batch_size", 8),
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        model, optimizer, train_dataloader, lr_scheduler = self._initialize_training(
            train_dataloader,
            num_epochs
        )
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            epoch_synthesis_quality = []
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                
                with self.accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        use_parallel_synthesis=True
                    )
                    
                    loss = outputs["loss"]
                    validation_weight = batch.get("validation_score", torch.ones_like(loss))
                    weighted_loss = loss * validation_weight.mean()
                    
                    self.accelerator.backward(weighted_loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            self.training_args.get("gradient_clipping", 1.0)
                        )
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                if outputs.get("synthesis_metrics"):
                    quality = self._compute_synthesis_quality(outputs["synthesis_metrics"])
                    epoch_synthesis_quality.append(quality)
                
                if global_step % self.training_args.get("logging_steps", 10) == 0:
                    self._log_metrics(
                        {
                            "train/loss": loss.item(),
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/synthesis_quality": quality if epoch_synthesis_quality else 0,
                            "train/global_step": global_step,
                        }
                    )
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                global_step += 1
                
                if global_step % self.training_args.get("save_steps", 500) == 0:
                    self._save_checkpoint(model, optimizer, lr_scheduler, global_step)
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            avg_synthesis_quality = np.mean(epoch_synthesis_quality) if epoch_synthesis_quality else 0
            
            print(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, Synthesis Quality: {avg_synthesis_quality:.4f}")
            
            if eval_dataloader:
                eval_loss, eval_metrics = self._evaluate(model, eval_dataloader)
                print(f"Eval Loss: {eval_loss:.4f}")
                
                self._log_metrics({
                    "eval/loss": eval_loss,
                    **{f"eval/{k}": v for k, v in eval_metrics.items()}
                })
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self._save_best_model(model)
        
        self._save_final_model(model)
        
        if self.training_args.get("use_wandb", True):
            wandb.finish()
    
    def _initialize_training(
        self,
        train_dataloader: DataLoader,
        num_epochs: int
    ) -> Tuple[nn.Module, Any, DataLoader, Any]:
        
        num_training_steps = len(train_dataloader) * num_epochs
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_args.get("learning_rate", 5e-5),
            weight_decay=self.training_args.get("weight_decay", 0.01)
        )
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_args.get("warmup_steps", 500),
            num_training_steps=num_training_steps
        )
        
        if self.training_args.get("use_deepspeed", True):
            model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, lr_scheduler
            )
        else:
            model = self.accelerator.prepare(self.model)
        
        return model, optimizer, train_dataloader, lr_scheduler
    
    def _evaluate(
        self,
        model: nn.Module,
        eval_dataloader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        
        model.eval()
        total_loss = 0
        total_confidence = 0
        total_synthesis_quality = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_parallel_synthesis=True
                )
                
                total_loss += outputs["loss"].item()
                total_confidence += outputs["confidence_scores"].mean().item()
                
                if outputs.get("synthesis_metrics"):
                    quality = self._compute_synthesis_quality(outputs["synthesis_metrics"])
                    total_synthesis_quality.append(quality)
        
        avg_loss = total_loss / len(eval_dataloader)
        avg_confidence = total_confidence / len(eval_dataloader)
        avg_synthesis_quality = np.mean(total_synthesis_quality) if total_synthesis_quality else 0
        
        metrics = {
            "confidence": avg_confidence,
            "synthesis_quality": avg_synthesis_quality,
        }
        
        return avg_loss, metrics
    
    def _compute_synthesis_quality(self, synthesis_metrics: List[Dict]) -> float:
        if not synthesis_metrics:
            return 0.0
        
        gate_scores = [m["gate_scores"].mean().item() for m in synthesis_metrics if "gate_scores" in m]
        divergences = [m["chain_divergence"].item() for m in synthesis_metrics if "chain_divergence" in m]
        
        gate_entropy = -sum(s * np.log(s + 1e-10) for s in gate_scores) / len(gate_scores) if gate_scores else 0
        avg_divergence = np.mean(divergences) if divergences else 0
        
        quality = (1 - gate_entropy) * 0.5 + (1 / (1 + avg_divergence)) * 0.5
        
        return quality
    
    def _log_metrics(self, metrics: Dict[str, float]):
        if self.training_args.get("use_wandb", True):
            wandb.log(metrics)
    
    def _save_checkpoint(self, model, optimizer, lr_scheduler, step):
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.save_state(checkpoint_dir)
        
        with open(checkpoint_dir / "trainer_state.json", "w") as f:
            json.dump({
                "step": step,
                "best_model_path": str(self.output_dir / "best_model"),
            }, f)
    
    def _save_best_model(self, model):
        best_model_dir = self.output_dir / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.save_model(model, best_model_dir)
        self.tokenizer.save_pretrained(best_model_dir)
        
        with open(best_model_dir / "dps_config.json", "w") as f:
            json.dump(self.model_config.to_dict(), f, indent=2)
    
    def _save_final_model(self, model):
        final_model_dir = self.output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.save_model(model, final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        
        with open(final_model_dir / "dps_config.json", "w") as f:
            json.dump(self.model_config.to_dict(), f, indent=2)


class ReasoningChainTrainer:
    def __init__(
        self,
        model: DPSModel,
        reasoning_chains: ParallelReasoningChains,
        validator: ScientificValidator
    ):
        self.model = model
        self.reasoning_chains = reasoning_chains
        self.validator = validator
        
    async def train_reasoning_step(
        self,
        input_text: str,
        target_output: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        
        self.reasoning_chains.initialize_chains(input_text)
        
        best_output = None
        best_score = 0
        
        for iteration in range(max_iterations):
            expansion_results = await self.reasoning_chains.parallel_expand(
                self._generate_expansion,
                depth=iteration
            )
            
            self.reasoning_chains.cross_pollinate_chains()
            
            synthesis = self.reasoning_chains.synthesize_chains()
            
            generated_output = self._generate_from_synthesis(synthesis)
            
            validation_result = self.validator.validate(
                content=generated_output,
                reasoning_type="systematic",
                evidence=synthesis.get("top_reasoning_paths", [])
            )
            
            if validation_result.confidence > best_score:
                best_score = validation_result.confidence
                best_output = generated_output
            
            if validation_result.confidence > 0.9:
                break
            
            for chain in self.reasoning_chains.chains:
                chain.prune_low_confidence_paths(threshold=0.3)
        
        return {
            "output": best_output,
            "score": best_score,
            "iterations": iteration + 1,
            "synthesis": synthesis,
        }
    
    async def _generate_expansion(self, node, chain_id):
        with torch.no_grad():
            input_text = node.content
            inputs = self.model.base_model.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            outputs = self.model.generate_with_synthesis(
                input_ids=inputs["input_ids"],
                max_new_tokens=128,
                temperature=0.7
            )
            
            generated_text = self.model.base_model.tokenizer.decode(
                outputs[0][0],
                skip_special_tokens=True
            )
            
        expansions = []
        for i in range(3):
            expansions.append({
                "content": f"{generated_text} [Branch {i+1}]",
                "type": np.random.choice(["DEDUCTIVE", "INDUCTIVE", "CAUSAL"]),
                "confidence": np.random.uniform(0.6, 0.95),
                "supporting": [f"Evidence {i+1}"],
                "contradicting": []
            })
        
        return expansions
    
    def _generate_from_synthesis(self, synthesis: Dict[str, Any]) -> str:
        top_paths = synthesis.get("top_reasoning_paths", [])
        if not top_paths:
            return "No valid reasoning path found"
        
        best_path = top_paths[0]
        return " -> ".join(best_path["path"])