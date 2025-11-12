from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


@dataclass
class DPSConfig:
    base_model_name: str = "meta-llama/Llama-3.1-70B"
    num_parallel_chains: int = 8
    reasoning_depth: int = 5
    hidden_size: int = 8192
    intermediate_size: int = 28672
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    num_hidden_layers: int = 80
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    use_flash_attention: bool = True
    synthesis_temperature: float = 0.7
    validation_threshold: float = 0.85
    gradient_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ParallelSynthesisLayer(nn.Module):
    def __init__(self, config: DPSConfig):
        super().__init__()
        self.config = config
        self.num_chains = config.num_parallel_chains

        self.chain_embeddings = nn.Parameter(
            torch.randn(self.num_chains, config.hidden_size) * 0.02
        )

        self.cross_chain_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.synthesis_mlp = nn.Sequential(
            nn.Linear(config.hidden_size * self.num_chains, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

        self.chain_gates = nn.Linear(config.hidden_size, self.num_chains)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, hidden_dim = hidden_states.shape

        chain_states = []
        for i in range(self.num_chains):
            chain_embedding = self.chain_embeddings[i].unsqueeze(0).unsqueeze(0)
            chain_embedding = chain_embedding.expand(batch_size, seq_len, -1)
            chain_input = hidden_states + chain_embedding

            chain_output, _ = self.cross_chain_attention(
                chain_input, chain_input, chain_input, attn_mask=attention_mask
            )
            chain_states.append(chain_output)

        chain_states_tensor = torch.stack(chain_states, dim=2)

        gate_scores = torch.softmax(self.chain_gates(hidden_states), dim=-1)
        gate_scores = gate_scores.unsqueeze(-1)

        weighted_chains = (chain_states_tensor * gate_scores).sum(dim=2)

        concat_chains = chain_states_tensor.reshape(batch_size, seq_len, -1)
        synthesized = self.synthesis_mlp(concat_chains)

        output = self.layer_norm(synthesized + weighted_chains)

        metrics = {
            "gate_scores": gate_scores.squeeze(-1).mean(dim=[0, 1]),
            "chain_divergence": self._compute_chain_divergence(chain_states_tensor),
        }

        return output, metrics

    def _compute_chain_divergence(self, chain_states: torch.Tensor) -> torch.Tensor:
        mean_state = chain_states.mean(dim=2, keepdim=True)
        divergences = torch.norm(chain_states - mean_state, dim=-1).mean(dim=[0, 1])
        return divergences.mean()


class DPSModel(PreTrainedModel):
    def __init__(self, config: DPSConfig):
        base_config = AutoConfig.from_pretrained(config.base_model_name)
        super().__init__(base_config)

        self.config = config
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=(
                "flash_attention_2" if config.use_flash_attention else "sdpa"
            ),
        )

        self.synthesis_layers = nn.ModuleList(
            [ParallelSynthesisLayer(config) for _ in range(config.reasoning_depth)]
        )

        self.reasoning_projector = nn.Linear(config.hidden_size, config.hidden_size)

        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self.synthesis_metrics: List[Dict[str, Any]] = []

        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_parallel_synthesis: bool = True,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:

        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = outputs.hidden_states[-1]

        if use_parallel_synthesis:
            synthesized_states = hidden_states
            layer_metrics = []

            for i, synthesis_layer in enumerate(self.synthesis_layers):
                synthesized_states, metrics = synthesis_layer(
                    synthesized_states, attention_mask=attention_mask
                )
                metrics["layer_idx"] = i
                layer_metrics.append(metrics)

            hidden_states = self.reasoning_projector(synthesized_states)
            self.synthesis_metrics = layer_metrics

        confidence_scores = self.confidence_head(hidden_states)

        lm_logits = self.base_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            confidence_weight = confidence_scores[..., :-1, :].squeeze(-1)
            weighted_loss = loss * confidence_weight.mean()
            loss = 0.8 * loss + 0.2 * weighted_loss

        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
            "confidence_scores": confidence_scores,
            "synthesis_metrics": self.synthesis_metrics,
        }

    def generate_with_synthesis(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        temperature = temperature or self.config.synthesis_temperature

        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids, use_parallel_synthesis=True, **kwargs
            )

            confidence_threshold = outputs["confidence_scores"].mean()

            if confidence_threshold < self.config.validation_threshold:
                temperature *= 1.2

            generated = self.base_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs,
            )

            generation_metrics = {
                "confidence_threshold": confidence_threshold.item(),
                "adjusted_temperature": temperature,
                "synthesis_depth": len(self.synthesis_metrics),
            }

        return generated, generation_metrics

    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)

        import json
        import os

        config_path = os.path.join(save_directory, "dps_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        import json
        import os

        config_path = os.path.join(model_path, "dps_config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = DPSConfig(**config_dict)
        model = cls(config)

        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"
        )
        model.load_state_dict(state_dict)

        return model
