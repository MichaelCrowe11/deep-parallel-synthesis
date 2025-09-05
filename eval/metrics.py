import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from dps.validator import ScientificValidator, ValidationStatus
from dps.reasoning import ReasoningType


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()


class ScientificReasoningMetrics:
    def __init__(self):
        self.validator = ScientificValidator()
        self.results_history = []
        
    def logical_consistency_score(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        scores = []
        details = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            pred_validation = self.validator.validate(
                content=pred,
                reasoning_type="deductive"
            )
            
            ref_validation = self.validator.validate(
                content=ref,
                reasoning_type="deductive"
            )
            
            consistency_score = self._compute_consistency(pred_validation, ref_validation)
            scores.append(consistency_score)
            
            details["individual_scores"].append(consistency_score)
            details["pred_confidence"].append(pred_validation.confidence)
            details["ref_confidence"].append(ref_validation.confidence)
        
        avg_score = np.mean(scores)
        
        result = EvaluationResult(
            metric_name="logical_consistency",
            score=avg_score,
            details={
                "mean": avg_score,
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                **details
            }
        )
        
        self.results_history.append(result)
        return result
    
    def _compute_consistency(self, pred_val, ref_val) -> float:
        status_match = 1.0 if pred_val.status == ref_val.status else 0.5
        
        confidence_diff = abs(pred_val.confidence - ref_val.confidence)
        confidence_similarity = 1.0 - confidence_diff
        
        evidence_overlap = len(set(pred_val.evidence) & set(ref_val.evidence))
        evidence_score = evidence_overlap / max(len(pred_val.evidence) + len(ref_val.evidence), 1)
        
        consistency = (
            status_match * 0.4 +
            confidence_similarity * 0.3 +
            evidence_score * 0.3
        )
        
        return consistency
    
    def reasoning_depth_score(self, reasoning_chains: List[List[str]]) -> EvaluationResult:
        depths = [len(chain) for chain in reasoning_chains]
        
        avg_depth = np.mean(depths)
        normalized_depth = min(avg_depth / 10.0, 1.0)
        
        depth_variance = np.var(depths)
        consistency_score = 1.0 / (1.0 + depth_variance)
        
        final_score = normalized_depth * 0.7 + consistency_score * 0.3
        
        result = EvaluationResult(
            metric_name="reasoning_depth",
            score=final_score,
            details={
                "average_depth": avg_depth,
                "max_depth": max(depths),
                "min_depth": min(depths),
                "depth_variance": depth_variance,
                "consistency": consistency_score
            }
        )
        
        self.results_history.append(result)
        return result
    
    def convergence_quality_score(
        self,
        convergence_points: List[Tuple[str, str]],
        total_chains: int
    ) -> EvaluationResult:
        
        if not convergence_points:
            return EvaluationResult(
                metric_name="convergence_quality",
                score=0.0,
                details={"message": "No convergence points found"}
            )
        
        convergence_ratio = len(convergence_points) / (total_chains * (total_chains - 1) / 2)
        
        unique_convergences = len(set(convergence_points))
        diversity_score = unique_convergences / max(len(convergence_points), 1)
        
        final_score = convergence_ratio * 0.6 + diversity_score * 0.4
        
        result = EvaluationResult(
            metric_name="convergence_quality",
            score=final_score,
            details={
                "num_convergences": len(convergence_points),
                "unique_convergences": unique_convergences,
                "convergence_ratio": convergence_ratio,
                "diversity_score": diversity_score
            }
        )
        
        self.results_history.append(result)
        return result
    
    def scientific_accuracy_score(
        self,
        predictions: List[str],
        references: List[str],
        domain: str = "general"
    ) -> EvaluationResult:
        
        domain_keywords = {
            "physics": ["force", "energy", "momentum", "quantum", "relativity"],
            "chemistry": ["molecule", "reaction", "bond", "element", "compound"],
            "biology": ["cell", "gene", "protein", "evolution", "organism"],
            "mathematics": ["theorem", "proof", "equation", "function", "integral"],
            "general": []
        }
        
        keywords = domain_keywords.get(domain, [])
        
        scores = []
        for pred, ref in zip(predictions, references):
            keyword_score = self._compute_keyword_overlap(pred, ref, keywords)
            
            structure_score = self._compute_structural_similarity(pred, ref)
            
            accuracy = keyword_score * 0.4 + structure_score * 0.6
            scores.append(accuracy)
        
        avg_score = np.mean(scores)
        
        result = EvaluationResult(
            metric_name="scientific_accuracy",
            score=avg_score,
            details={
                "domain": domain,
                "mean_accuracy": avg_score,
                "std": np.std(scores),
                "num_samples": len(predictions)
            }
        )
        
        self.results_history.append(result)
        return result
    
    def _compute_keyword_overlap(self, pred: str, ref: str, keywords: List[str]) -> float:
        if not keywords:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, pred.lower(), ref.lower()).ratio()
        
        pred_keywords = sum(1 for kw in keywords if kw in pred.lower())
        ref_keywords = sum(1 for kw in keywords if kw in ref.lower())
        
        if ref_keywords == 0:
            return 1.0 if pred_keywords == 0 else 0.0
        
        return min(pred_keywords / ref_keywords, 1.0)
    
    def _compute_structural_similarity(self, pred: str, ref: str) -> float:
        pred_sentences = pred.split('.')
        ref_sentences = ref.split('.')
        
        len_similarity = 1.0 - abs(len(pred_sentences) - len(ref_sentences)) / max(len(ref_sentences), 1)
        
        pred_has_conclusion = any(word in pred.lower() for word in ["therefore", "thus", "conclude"])
        ref_has_conclusion = any(word in ref.lower() for word in ["therefore", "thus", "conclude"])
        conclusion_match = 1.0 if pred_has_conclusion == ref_has_conclusion else 0.5
        
        return len_similarity * 0.5 + conclusion_match * 0.5


class ReasoningTypeClassifier:
    def __init__(self):
        self.reasoning_types = list(ReasoningType)
        self.type_patterns = {
            ReasoningType.DEDUCTIVE: ["if", "then", "therefore", "thus", "implies"],
            ReasoningType.INDUCTIVE: ["observe", "pattern", "generally", "data shows"],
            ReasoningType.ABDUCTIVE: ["best explanation", "hypothesis", "likely because"],
            ReasoningType.CAUSAL: ["causes", "leads to", "results in", "effect"],
            ReasoningType.PROBABILISTIC: ["probability", "likely", "chance", "odds"],
            ReasoningType.ANALOGICAL: ["similar to", "like", "analogous", "comparison"],
            ReasoningType.COUNTERFACTUAL: ["what if", "would have", "alternative"],
            ReasoningType.SYSTEMATIC: ["step by step", "systematic", "comprehensive"]
        }
    
    def classify(self, text: str) -> ReasoningType:
        scores = {}
        
        for rtype, patterns in self.type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text.lower())
            scores[rtype] = score
        
        if max(scores.values()) == 0:
            return ReasoningType.SYSTEMATIC
        
        return max(scores, key=scores.get)
    
    def evaluate_classification(
        self,
        predictions: List[str],
        true_types: List[ReasoningType]
    ) -> Dict[str, Any]:
        
        predicted_types = [self.classify(pred) for pred in predictions]
        
        accuracy = accuracy_score(
            [t.value for t in true_types],
            [t.value for t in predicted_types]
        )
        
        precision, recall, f1, support = precision_recall_fscore_support(
            [t.value for t in true_types],
            [t.value for t in predicted_types],
            average='weighted',
            zero_division=0
        )
        
        conf_matrix = confusion_matrix(
            [t.value for t in true_types],
            [t.value for t in predicted_types],
            labels=[t.value for t in self.reasoning_types]
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "support": support
        }


class DPSEvaluator:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.metrics = ScientificReasoningMetrics()
        self.classifier = ReasoningTypeClassifier()
        self.results = {}
        
    def evaluate_dataset(
        self,
        dataset_path: str,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        predictions = []
        references = []
        reasoning_chains = []
        
        for item in tqdm(dataset, desc="Evaluating"):
            pred = self._generate_prediction(item["prompt"])
            predictions.append(pred["text"])
            references.append(item["response"])
            
            if "reasoning_chain" in pred:
                reasoning_chains.append(pred["reasoning_chain"])
        
        consistency_result = self.metrics.logical_consistency_score(predictions, references)
        self.results["logical_consistency"] = consistency_result
        
        if reasoning_chains:
            depth_result = self.metrics.reasoning_depth_score(reasoning_chains)
            self.results["reasoning_depth"] = depth_result
        
        accuracy_result = self.metrics.scientific_accuracy_score(predictions, references)
        self.results["scientific_accuracy"] = accuracy_result
        
        if "reasoning_types" in dataset[0]:
            true_types = [ReasoningType[item["reasoning_type"]] for item in dataset]
            classification_results = self.classifier.evaluate_classification(predictions, true_types)
            self.results["reasoning_classification"] = classification_results
        
        self._save_results(output_dir)
        self._generate_visualizations(output_dir)
        
        return self.results
    
    def _generate_prediction(self, prompt: str) -> Dict[str, Any]:
        return {
            "text": f"Generated response for: {prompt}",
            "reasoning_chain": ["Step 1", "Step 2", "Step 3"]
        }
    
    def _save_results(self, output_dir: Path):
        results_file = output_dir / "evaluation_results.json"
        
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, EvaluationResult):
                serializable_results[key] = {
                    "score": value.score,
                    "details": value.details,
                    "timestamp": value.timestamp
                }
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def _generate_visualizations(self, output_dir: Path):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if "logical_consistency" in self.results:
            result = self.results["logical_consistency"]
            axes[0, 0].bar(["Logical Consistency"], [result.score])
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_title("Logical Consistency Score")
            axes[0, 0].set_ylabel("Score")
        
        if "reasoning_depth" in self.results:
            result = self.results["reasoning_depth"]
            axes[0, 1].bar(["Reasoning Depth"], [result.score])
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].set_title("Reasoning Depth Score")
            axes[0, 1].set_ylabel("Score")
        
        if "scientific_accuracy" in self.results:
            result = self.results["scientific_accuracy"]
            axes[1, 0].bar(["Scientific Accuracy"], [result.score])
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title("Scientific Accuracy Score")
            axes[1, 0].set_ylabel("Score")
        
        if "reasoning_classification" in self.results:
            classification = self.results["reasoning_classification"]
            metrics = ["Accuracy", "Precision", "Recall", "F1"]
            scores = [
                classification["accuracy"],
                classification["precision"],
                classification["recall"],
                classification["f1_score"]
            ]
            axes[1, 1].bar(metrics, scores)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title("Reasoning Type Classification")
            axes[1, 1].set_ylabel("Score")
        
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        if "reasoning_classification" in self.results and "confusion_matrix" in self.results["reasoning_classification"]:
            conf_matrix = np.array(self.results["reasoning_classification"]["confusion_matrix"])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[t.value for t in ReasoningType],
                yticklabels=[t.value for t in ReasoningType]
            )
            plt.title("Reasoning Type Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def generate_report(self, output_dir: str = "./evaluation_results") -> str:
        output_dir = Path(output_dir)
        
        report = []
        report.append("# Deep Parallel Synthesis Evaluation Report\n")
        report.append(f"Model: {self.model_path}\n")
        report.append(f"Timestamp: {self.metrics.results_history[0].timestamp if self.metrics.results_history else 'N/A'}\n\n")
        
        report.append("## Summary Metrics\n\n")
        
        if self.results:
            for metric_name, result in self.results.items():
                if isinstance(result, EvaluationResult):
                    report.append(f"### {metric_name.replace('_', ' ').title()}\n")
                    report.append(f"- **Score**: {result.score:.4f}\n")
                    if isinstance(result.details, dict):
                        for key, value in result.details.items():
                            if isinstance(value, (int, float)):
                                report.append(f"- {key}: {value:.4f}\n")
                    report.append("\n")
                elif isinstance(result, dict) and "accuracy" in result:
                    report.append(f"### {metric_name.replace('_', ' ').title()}\n")
                    report.append(f"- **Accuracy**: {result['accuracy']:.4f}\n")
                    report.append(f"- **Precision**: {result.get('precision', 0):.4f}\n")
                    report.append(f"- **Recall**: {result.get('recall', 0):.4f}\n")
                    report.append(f"- **F1 Score**: {result.get('f1_score', 0):.4f}\n")
                    report.append("\n")
        
        report.append("## Visualization Files\n\n")
        report.append("- `evaluation_metrics.png`: Overall metrics visualization\n")
        report.append("- `confusion_matrix.png`: Reasoning type classification confusion matrix\n")
        report.append("- `evaluation_results.json`: Detailed results in JSON format\n")
        
        report_text = "".join(report)
        
        report_file = output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to {report_file}")
        
        return report_text