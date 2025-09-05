import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import asyncio
import concurrent.futures


class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    ANALOGICAL = "analogical"
    COUNTERFACTUAL = "counterfactual"
    SYSTEMATIC = "systematic"


@dataclass
class ReasoningNode:
    node_id: str
    reasoning_type: ReasoningType
    content: str
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    children: List['ReasoningNode']
    parent: Optional['ReasoningNode'] = None
    depth: int = 0
    validation_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def add_child(self, child: 'ReasoningNode'):
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def get_path_to_root(self) -> List['ReasoningNode']:
        path = [self]
        current = self.parent
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def compute_aggregate_confidence(self) -> float:
        if not self.children:
            return self.confidence
        
        child_confidences = [child.compute_aggregate_confidence() for child in self.children]
        weighted_confidence = self.confidence * 0.6 + np.mean(child_confidences) * 0.4
        return min(weighted_confidence, 1.0)


class ReasoningChain:
    def __init__(self, chain_id: str, initial_premise: str):
        self.chain_id = chain_id
        self.root = ReasoningNode(
            node_id=f"{chain_id}_root",
            reasoning_type=ReasoningType.SYSTEMATIC,
            content=initial_premise,
            confidence=1.0,
            supporting_evidence=[],
            contradicting_evidence=[],
            children=[]
        )
        self.current_frontier = [self.root]
        self.all_nodes = {self.root.node_id: self.root}
        self.convergence_points = []
        
    def expand_node(
        self,
        node: ReasoningNode,
        expansions: List[Dict[str, Any]]
    ) -> List[ReasoningNode]:
        new_nodes = []
        for expansion in expansions:
            new_node = ReasoningNode(
                node_id=f"{self.chain_id}_{len(self.all_nodes)}",
                reasoning_type=ReasoningType[expansion.get("type", "DEDUCTIVE")],
                content=expansion["content"],
                confidence=expansion.get("confidence", 0.8),
                supporting_evidence=expansion.get("supporting", []),
                contradicting_evidence=expansion.get("contradicting", []),
                children=[]
            )
            node.add_child(new_node)
            self.all_nodes[new_node.node_id] = new_node
            new_nodes.append(new_node)
        return new_nodes
    
    def prune_low_confidence_paths(self, threshold: float = 0.3):
        to_remove = []
        for node_id, node in self.all_nodes.items():
            if node.compute_aggregate_confidence() < threshold and node != self.root:
                to_remove.append(node_id)
        
        for node_id in to_remove:
            node = self.all_nodes[node_id]
            if node.parent:
                node.parent.children.remove(node)
            del self.all_nodes[node_id]
    
    def find_convergence_points(self, other_chain: 'ReasoningChain') -> List[Tuple[ReasoningNode, ReasoningNode]]:
        convergences = []
        for node1 in self.all_nodes.values():
            for node2 in other_chain.all_nodes.values():
                similarity = self._compute_node_similarity(node1, node2)
                if similarity > 0.85:
                    convergences.append((node1, node2))
        return convergences
    
    def _compute_node_similarity(self, node1: ReasoningNode, node2: ReasoningNode) -> float:
        from difflib import SequenceMatcher
        
        content_similarity = SequenceMatcher(None, node1.content, node2.content).ratio()
        
        type_similarity = 1.0 if node1.reasoning_type == node2.reasoning_type else 0.5
        
        confidence_similarity = 1.0 - abs(node1.confidence - node2.confidence)
        
        evidence_overlap = len(set(node1.supporting_evidence) & set(node2.supporting_evidence))
        evidence_similarity = evidence_overlap / max(
            len(node1.supporting_evidence) + len(node2.supporting_evidence), 1
        )
        
        similarity = (
            content_similarity * 0.4 +
            type_similarity * 0.2 +
            confidence_similarity * 0.2 +
            evidence_similarity * 0.2
        )
        
        return similarity


class ParallelReasoningChains:
    def __init__(
        self,
        num_chains: int = 8,
        max_depth: int = 5,
        expansion_factor: int = 3,
        convergence_threshold: float = 0.85,
        device: str = "cuda"
    ):
        self.num_chains = num_chains
        self.max_depth = max_depth
        self.expansion_factor = expansion_factor
        self.convergence_threshold = convergence_threshold
        self.device = device
        
        self.chains: List[ReasoningChain] = []
        self.global_knowledge_graph = {}
        self.synthesis_history = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_chains)
        
    def initialize_chains(self, initial_premise: str) -> None:
        self.chains = [
            ReasoningChain(
                chain_id=f"chain_{i}",
                initial_premise=initial_premise
            )
            for i in range(self.num_chains)
        ]
    
    async def parallel_expand(
        self,
        expansion_function,
        depth: int
    ) -> List[List[ReasoningNode]]:
        
        async def expand_chain(chain: ReasoningChain) -> List[ReasoningNode]:
            new_frontier = []
            for node in chain.current_frontier:
                if node.depth < self.max_depth:
                    expansions = await expansion_function(node, chain.chain_id)
                    new_nodes = chain.expand_node(node, expansions)
                    new_frontier.extend(new_nodes)
            chain.current_frontier = new_frontier
            return new_frontier
        
        tasks = [expand_chain(chain) for chain in self.chains]
        results = await asyncio.gather(*tasks)
        return results
    
    def cross_pollinate_chains(self) -> Dict[str, List[Tuple[str, str]]]:
        cross_pollination_map = {}
        
        for i, chain1 in enumerate(self.chains):
            for j, chain2 in enumerate(self.chains):
                if i < j:
                    convergences = chain1.find_convergence_points(chain2)
                    if convergences:
                        key = f"{chain1.chain_id}_{chain2.chain_id}"
                        cross_pollination_map[key] = [
                            (n1.node_id, n2.node_id) for n1, n2 in convergences
                        ]
                        
                        for node1, node2 in convergences:
                            merged_evidence = list(set(
                                node1.supporting_evidence + node2.supporting_evidence
                            ))
                            node1.supporting_evidence = merged_evidence
                            node2.supporting_evidence = merged_evidence
                            
                            avg_confidence = (node1.confidence + node2.confidence) / 2
                            node1.confidence = avg_confidence
                            node2.confidence = avg_confidence
        
        return cross_pollination_map
    
    def synthesize_chains(self) -> Dict[str, Any]:
        all_leaf_nodes = []
        for chain in self.chains:
            leaf_nodes = [
                node for node in chain.all_nodes.values()
                if not node.children
            ]
            all_leaf_nodes.extend(leaf_nodes)
        
        high_confidence_nodes = [
            node for node in all_leaf_nodes
            if node.compute_aggregate_confidence() > 0.7
        ]
        
        reasoning_paths = []
        for node in high_confidence_nodes:
            path = node.get_path_to_root()
            reasoning_paths.append({
                "path": [n.content for n in path],
                "types": [n.reasoning_type.value for n in path],
                "confidence": node.compute_aggregate_confidence(),
                "evidence": node.supporting_evidence
            })
        
        reasoning_paths.sort(key=lambda x: x["confidence"], reverse=True)
        
        reasoning_type_distribution = {}
        for chain in self.chains:
            for node in chain.all_nodes.values():
                rt = node.reasoning_type.value
                reasoning_type_distribution[rt] = reasoning_type_distribution.get(rt, 0) + 1
        
        convergence_strength = self._compute_convergence_strength()
        
        synthesis = {
            "top_reasoning_paths": reasoning_paths[:5],
            "num_total_nodes": sum(len(chain.all_nodes) for chain in self.chains),
            "num_convergence_points": convergence_strength["num_convergences"],
            "average_confidence": np.mean([
                node.confidence for chain in self.chains
                for node in chain.all_nodes.values()
            ]),
            "reasoning_type_distribution": reasoning_type_distribution,
            "convergence_strength": convergence_strength["strength"],
            "synthesis_quality": self._compute_synthesis_quality(),
        }
        
        self.synthesis_history.append(synthesis)
        return synthesis
    
    def _compute_convergence_strength(self) -> Dict[str, Any]:
        total_convergences = 0
        convergence_confidences = []
        
        for i, chain1 in enumerate(self.chains):
            for j, chain2 in enumerate(self.chains):
                if i < j:
                    convergences = chain1.find_convergence_points(chain2)
                    total_convergences += len(convergences)
                    for node1, node2 in convergences:
                        avg_conf = (node1.confidence + node2.confidence) / 2
                        convergence_confidences.append(avg_conf)
        
        strength = 0.0
        if convergence_confidences:
            strength = np.mean(convergence_confidences) * (total_convergences / (self.num_chains * 2))
        
        return {
            "num_convergences": total_convergences,
            "strength": min(strength, 1.0),
            "avg_convergence_confidence": np.mean(convergence_confidences) if convergence_confidences else 0.0
        }
    
    def _compute_synthesis_quality(self) -> float:
        diversity_score = self._compute_diversity_score()
        
        convergence_data = self._compute_convergence_strength()
        convergence_score = convergence_data["strength"]
        
        avg_depth = np.mean([
            max(node.depth for node in chain.all_nodes.values())
            for chain in self.chains
        ]) / self.max_depth
        
        avg_confidence = np.mean([
            node.confidence for chain in self.chains
            for node in chain.all_nodes.values()
        ])
        
        quality = (
            diversity_score * 0.25 +
            convergence_score * 0.35 +
            avg_depth * 0.2 +
            avg_confidence * 0.2
        )
        
        return min(quality, 1.0)
    
    def _compute_diversity_score(self) -> float:
        reasoning_types_per_chain = []
        for chain in self.chains:
            types = set(node.reasoning_type for node in chain.all_nodes.values())
            reasoning_types_per_chain.append(len(types))
        
        diversity = np.mean(reasoning_types_per_chain) / len(ReasoningType)
        return diversity
    
    def get_best_reasoning_path(self) -> Optional[List[str]]:
        best_path = None
        best_confidence = 0.0
        
        for chain in self.chains:
            for node in chain.all_nodes.values():
                if not node.children:
                    confidence = node.compute_aggregate_confidence()
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_path = [n.content for n in node.get_path_to_root()]
        
        return best_path
    
    def visualize_chains(self) -> str:
        visualization = []
        visualization.append(f"Parallel Reasoning Chains ({self.num_chains} chains)\n")
        visualization.append("=" * 50 + "\n")
        
        for chain in self.chains:
            visualization.append(f"\n{chain.chain_id}:\n")
            self._visualize_chain_recursive(chain.root, visualization, prefix="")
        
        convergence_data = self._compute_convergence_strength()
        visualization.append(f"\nConvergence Points: {convergence_data['num_convergences']}\n")
        visualization.append(f"Convergence Strength: {convergence_data['strength']:.2f}\n")
        visualization.append(f"Synthesis Quality: {self._compute_synthesis_quality():.2f}\n")
        
        return "".join(visualization)
    
    def _visualize_chain_recursive(
        self,
        node: ReasoningNode,
        visualization: List[str],
        prefix: str,
        is_last: bool = True
    ):
        connector = "└── " if is_last else "├── "
        visualization.append(f"{prefix}{connector}[{node.reasoning_type.value[:3]}] ")
        visualization.append(f"{node.content[:50]}... (conf: {node.confidence:.2f})\n")
        
        extension = "    " if is_last else "│   "
        for i, child in enumerate(node.children):
            self._visualize_chain_recursive(
                child,
                visualization,
                prefix + extension,
                i == len(node.children) - 1
            )
    
    def export_to_graph(self) -> Dict[str, Any]:
        nodes = []
        edges = []
        
        for chain in self.chains:
            for node in chain.all_nodes.values():
                nodes.append({
                    "id": node.node_id,
                    "chain": chain.chain_id,
                    "content": node.content,
                    "type": node.reasoning_type.value,
                    "confidence": node.confidence,
                    "depth": node.depth
                })
                
                for child in node.children:
                    edges.append({
                        "source": node.node_id,
                        "target": child.node_id,
                        "weight": child.confidence
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "num_chains": self.num_chains,
                "max_depth": self.max_depth,
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            }
        }