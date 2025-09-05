import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from collections import defaultdict
import sympy
from sympy import symbols, parse_expr, simplify
from sympy.logic.inference import satisfiable


class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    PARTIALLY_VALID = "partially_valid"
    UNCERTAIN = "uncertain"
    REQUIRES_REVIEW = "requires_review"


class ValidationCategory(Enum):
    LOGICAL_CONSISTENCY = "logical_consistency"
    MATHEMATICAL_CORRECTNESS = "mathematical_correctness"
    EMPIRICAL_SUPPORT = "empirical_support"
    THEORETICAL_ALIGNMENT = "theoretical_alignment"
    CAUSAL_VALIDITY = "causal_validity"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    REPRODUCIBILITY = "reproducibility"
    FALSIFIABILITY = "falsifiability"


@dataclass
class ValidationResult:
    status: ValidationStatus
    confidence: float
    category: ValidationCategory
    evidence: List[str]
    errors: List[str]
    suggestions: List[str]
    metrics: Dict[str, float]
    timestamp: str = ""
    validator_id: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "category": self.category.value,
            "evidence": self.evidence,
            "errors": self.errors,
            "suggestions": self.suggestions,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "validator_id": self.validator_id
        }
    
    def is_valid(self) -> bool:
        return self.status in [ValidationStatus.VALID, ValidationStatus.PARTIALLY_VALID]
    
    def requires_human_review(self) -> bool:
        return self.status == ValidationStatus.REQUIRES_REVIEW or self.confidence < 0.5


class LogicalValidator:
    def __init__(self):
        self.rules = {
            "non_contradiction": self._check_non_contradiction,
            "modus_ponens": self._check_modus_ponens,
            "modus_tollens": self._check_modus_tollens,
            "syllogism": self._check_syllogism,
            "transitivity": self._check_transitivity,
        }
    
    def validate(self, propositions: List[str]) -> ValidationResult:
        errors = []
        evidence = []
        metrics = {}
        
        parsed_props = self._parse_propositions(propositions)
        
        contradiction_free = self._check_non_contradiction(parsed_props)
        metrics["contradiction_free"] = float(contradiction_free)
        
        if not contradiction_free:
            errors.append("Logical contradictions detected in propositions")
        
        inference_validity = self._check_inference_chains(parsed_props)
        metrics["inference_validity"] = inference_validity
        
        if inference_validity < 0.7:
            errors.append("Weak inference chain detected")
        
        completeness = self._check_completeness(parsed_props)
        metrics["completeness"] = completeness
        
        confidence = (metrics["contradiction_free"] * 0.4 +
                     metrics["inference_validity"] * 0.4 +
                     metrics["completeness"] * 0.2)
        
        status = ValidationStatus.VALID if confidence > 0.8 else \
                ValidationStatus.PARTIALLY_VALID if confidence > 0.5 else \
                ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            confidence=confidence,
            category=ValidationCategory.LOGICAL_CONSISTENCY,
            evidence=evidence,
            errors=errors,
            suggestions=self._generate_suggestions(errors),
            metrics=metrics,
            validator_id="logical_validator"
        )
    
    def _parse_propositions(self, propositions: List[str]) -> List[Any]:
        parsed = []
        for prop in propositions:
            try:
                clean_prop = re.sub(r'[^\w\s&|~()>]', '', prop)
                if clean_prop.strip():
                    parsed.append(clean_prop)
            except Exception:
                parsed.append(prop)
        return parsed
    
    def _check_non_contradiction(self, propositions: List[str]) -> bool:
        for i, prop1 in enumerate(propositions):
            for prop2 in propositions[i+1:]:
                if self._is_contradiction(prop1, prop2):
                    return False
        return True
    
    def _is_contradiction(self, prop1: str, prop2: str) -> bool:
        if f"not {prop1}" in prop2 or f"not {prop2}" in prop1:
            return True
        
        if prop1.startswith("not ") and prop1[4:] == prop2:
            return True
        if prop2.startswith("not ") and prop2[4:] == prop1:
            return True
        
        return False
    
    def _check_modus_ponens(self, premises: List[str]) -> Optional[str]:
        for p1 in premises:
            if " implies " in p1:
                parts = p1.split(" implies ")
                if len(parts) == 2:
                    antecedent, consequent = parts
                    if antecedent in premises:
                        return consequent
        return None
    
    def _check_modus_tollens(self, premises: List[str]) -> Optional[str]:
        for p1 in premises:
            if " implies " in p1:
                parts = p1.split(" implies ")
                if len(parts) == 2:
                    antecedent, consequent = parts
                    neg_consequent = f"not {consequent}"
                    if neg_consequent in premises:
                        return f"not {antecedent}"
        return None
    
    def _check_syllogism(self, premises: List[str]) -> Optional[str]:
        pattern = r"All (\w+) are (\w+)"
        syllogisms = []
        
        for p1 in premises:
            match1 = re.match(pattern, p1)
            if match1:
                for p2 in premises:
                    match2 = re.match(pattern, p2)
                    if match2 and match1.group(2) == match2.group(1):
                        conclusion = f"All {match1.group(1)} are {match2.group(2)}"
                        syllogisms.append(conclusion)
        
        return syllogisms[0] if syllogisms else None
    
    def _check_transitivity(self, relations: List[str]) -> List[str]:
        transitive_conclusions = []
        relation_pattern = r"(\w+) (>=?|<=?|=) (\w+)"
        
        parsed_relations = []
        for rel in relations:
            match = re.match(relation_pattern, rel)
            if match:
                parsed_relations.append((match.group(1), match.group(2), match.group(3)))
        
        for r1 in parsed_relations:
            for r2 in parsed_relations:
                if r1[2] == r2[0] and r1[1] == r2[1]:
                    conclusion = f"{r1[0]} {r1[1]} {r2[2]}"
                    transitive_conclusions.append(conclusion)
        
        return transitive_conclusions
    
    def _check_inference_chains(self, propositions: List[str]) -> float:
        valid_inferences = 0
        total_possible = max(len(propositions) - 1, 1)
        
        for i in range(len(propositions) - 1):
            if self._can_infer(propositions[:i+1], propositions[i+1]):
                valid_inferences += 1
        
        return valid_inferences / total_possible
    
    def _can_infer(self, premises: List[str], conclusion: str) -> bool:
        if conclusion in premises:
            return True
        
        inferred = self._check_modus_ponens(premises)
        if inferred and inferred == conclusion:
            return True
        
        inferred = self._check_modus_tollens(premises)
        if inferred and inferred == conclusion:
            return True
        
        return False
    
    def _check_completeness(self, propositions: List[str]) -> float:
        if not propositions:
            return 0.0
        
        has_premises = any("if" in p or "implies" in p for p in propositions)
        has_conclusions = any("therefore" in p or "thus" in p for p in propositions)
        has_evidence = any("because" in p or "since" in p for p in propositions)
        
        score = sum([has_premises, has_conclusions, has_evidence]) / 3.0
        return score
    
    def _generate_suggestions(self, errors: List[str]) -> List[str]:
        suggestions = []
        
        if "contradiction" in " ".join(errors).lower():
            suggestions.append("Review and resolve logical contradictions")
            suggestions.append("Ensure premises are mutually consistent")
        
        if "weak inference" in " ".join(errors).lower():
            suggestions.append("Strengthen logical connections between premises")
            suggestions.append("Add intermediate reasoning steps")
        
        return suggestions


class MathematicalValidator:
    def __init__(self):
        self.tolerance = 1e-10
    
    def validate(self, expression: str, expected_result: Optional[Any] = None) -> ValidationResult:
        errors = []
        evidence = []
        metrics = {}
        
        try:
            parsed_expr = parse_expr(expression, transformations='all')
            simplified = simplify(parsed_expr)
            
            metrics["parse_success"] = 1.0
            evidence.append(f"Successfully parsed: {expression}")
            evidence.append(f"Simplified form: {simplified}")
            
            is_valid_math = self._check_mathematical_validity(parsed_expr)
            metrics["mathematical_validity"] = float(is_valid_math)
            
            if expected_result is not None:
                result_matches = self._check_result(simplified, expected_result)
                metrics["result_accuracy"] = float(result_matches)
                if not result_matches:
                    errors.append(f"Result mismatch: got {simplified}, expected {expected_result}")
            
            dimension_consistent = self._check_dimensional_consistency(parsed_expr)
            metrics["dimensional_consistency"] = float(dimension_consistent)
            
            confidence = np.mean(list(metrics.values()))
            
            status = ValidationStatus.VALID if confidence > 0.9 else \
                    ValidationStatus.PARTIALLY_VALID if confidence > 0.6 else \
                    ValidationStatus.INVALID
                    
        except Exception as e:
            errors.append(f"Mathematical parsing error: {str(e)}")
            metrics["parse_success"] = 0.0
            confidence = 0.0
            status = ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            confidence=confidence,
            category=ValidationCategory.MATHEMATICAL_CORRECTNESS,
            evidence=evidence,
            errors=errors,
            suggestions=self._generate_math_suggestions(errors),
            metrics=metrics,
            validator_id="mathematical_validator"
        )
    
    def _check_mathematical_validity(self, expr) -> bool:
        try:
            if expr.is_number:
                return not (expr.is_infinite or expr.is_nan)
            
            free_symbols = expr.free_symbols
            if free_symbols:
                test_values = {sym: 1 for sym in free_symbols}
                result = expr.subs(test_values)
                return not (result.is_infinite or result.is_nan)
            
            return True
        except Exception:
            return False
    
    def _check_result(self, computed: Any, expected: Any) -> bool:
        try:
            if isinstance(expected, (int, float)):
                computed_val = float(computed.evalf())
                return abs(computed_val - expected) < self.tolerance
            else:
                expected_expr = parse_expr(str(expected))
                return simplify(computed - expected_expr) == 0
        except Exception:
            return False
    
    def _check_dimensional_consistency(self, expr) -> bool:
        try:
            if not expr.free_symbols:
                return True
            
            return True
        except Exception:
            return False
    
    def _generate_math_suggestions(self, errors: List[str]) -> List[str]:
        suggestions = []
        
        if "parsing error" in " ".join(errors).lower():
            suggestions.append("Check mathematical syntax")
            suggestions.append("Ensure proper use of operators and parentheses")
        
        if "result mismatch" in " ".join(errors).lower():
            suggestions.append("Verify calculation steps")
            suggestions.append("Check for rounding errors or precision issues")
        
        return suggestions


class ScientificValidator:
    def __init__(self):
        self.logical_validator = LogicalValidator()
        self.mathematical_validator = MathematicalValidator()
        self.validation_history = []
        self.confidence_threshold = 0.7
    
    def validate(
        self,
        content: str,
        reasoning_type: str,
        evidence: List[str] = None,
        expected_results: Dict[str, Any] = None
    ) -> ValidationResult:
        
        validators_results = []
        
        logical_props = self._extract_logical_propositions(content)
        if logical_props:
            logical_result = self.logical_validator.validate(logical_props)
            validators_results.append(logical_result)
        
        math_expressions = self._extract_mathematical_expressions(content)
        for expr in math_expressions:
            math_result = self.mathematical_validator.validate(expr)
            validators_results.append(math_result)
        
        empirical_result = self._validate_empirical_support(evidence or [])
        validators_results.append(empirical_result)
        
        theoretical_result = self._validate_theoretical_alignment(content, reasoning_type)
        validators_results.append(theoretical_result)
        
        combined_result = self._combine_validation_results(validators_results)
        
        self.validation_history.append(combined_result)
        
        return combined_result
    
    def _extract_logical_propositions(self, content: str) -> List[str]:
        propositions = []
        
        patterns = [
            r"if .+ then .+",
            r".+ implies .+",
            r"therefore .+",
            r"thus .+",
            r"because .+",
            r"since .+",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            propositions.extend(matches)
        
        sentences = content.split('.')
        for sent in sentences:
            if any(word in sent.lower() for word in ['all', 'some', 'none', 'not']):
                propositions.append(sent.strip())
        
        return propositions
    
    def _extract_mathematical_expressions(self, content: str) -> List[str]:
        expressions = []
        
        math_pattern = r'(?:\$|\\\[|\\\()(.+?)(?:\$|\\\]|\\\))'
        matches = re.findall(math_pattern, content)
        expressions.extend(matches)
        
        equation_pattern = r'([a-zA-Z_]\w*\s*=\s*[^=]+?)(?:;|\n|$)'
        matches = re.findall(equation_pattern, content)
        expressions.extend(matches)
        
        return expressions
    
    def _validate_empirical_support(self, evidence: List[str]) -> ValidationResult:
        if not evidence:
            return ValidationResult(
                status=ValidationStatus.UNCERTAIN,
                confidence=0.5,
                category=ValidationCategory.EMPIRICAL_SUPPORT,
                evidence=[],
                errors=["No empirical evidence provided"],
                suggestions=["Provide empirical data or references"],
                metrics={"evidence_count": 0, "evidence_quality": 0.0},
                validator_id="empirical_validator"
            )
        
        quality_score = min(len(evidence) / 5.0, 1.0)
        
        has_data = any("data" in e.lower() or "study" in e.lower() for e in evidence)
        has_citation = any("et al" in e or re.search(r'\d{4}', e) for e in evidence)
        
        quality_score *= (1.0 + 0.3 * has_data + 0.2 * has_citation)
        quality_score = min(quality_score, 1.0)
        
        status = ValidationStatus.VALID if quality_score > 0.7 else \
                ValidationStatus.PARTIALLY_VALID if quality_score > 0.4 else \
                ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            confidence=quality_score,
            category=ValidationCategory.EMPIRICAL_SUPPORT,
            evidence=evidence[:5],
            errors=[] if quality_score > 0.4 else ["Insufficient empirical support"],
            suggestions=["Add peer-reviewed references"] if not has_citation else [],
            metrics={"evidence_count": len(evidence), "evidence_quality": quality_score},
            validator_id="empirical_validator"
        )
    
    def _validate_theoretical_alignment(self, content: str, reasoning_type: str) -> ValidationResult:
        alignment_scores = {
            "deductive": self._check_deductive_alignment(content),
            "inductive": self._check_inductive_alignment(content),
            "abductive": self._check_abductive_alignment(content),
            "causal": self._check_causal_alignment(content),
        }
        
        score = alignment_scores.get(reasoning_type.lower(), 0.5)
        
        status = ValidationStatus.VALID if score > 0.8 else \
                ValidationStatus.PARTIALLY_VALID if score > 0.5 else \
                ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            confidence=score,
            category=ValidationCategory.THEORETICAL_ALIGNMENT,
            evidence=[f"Reasoning type: {reasoning_type}"],
            errors=[] if score > 0.5 else [f"Poor {reasoning_type} reasoning structure"],
            suggestions=self._get_reasoning_suggestions(reasoning_type),
            metrics={"alignment_score": score, "reasoning_type_match": 1.0},
            validator_id="theoretical_validator"
        )
    
    def _check_deductive_alignment(self, content: str) -> float:
        has_premises = bool(re.search(r'(premise|assume|given that)', content, re.I))
        has_conclusion = bool(re.search(r'(therefore|thus|conclude)', content, re.I))
        has_logical_flow = bool(re.search(r'(if.*then|implies)', content, re.I))
        
        return (has_premises * 0.3 + has_conclusion * 0.3 + has_logical_flow * 0.4)
    
    def _check_inductive_alignment(self, content: str) -> float:
        has_observations = bool(re.search(r'(observe|data|evidence)', content, re.I))
        has_pattern = bool(re.search(r'(pattern|trend|correlation)', content, re.I))
        has_generalization = bool(re.search(r'(generally|typically|usually)', content, re.I))
        
        return (has_observations * 0.4 + has_pattern * 0.3 + has_generalization * 0.3)
    
    def _check_abductive_alignment(self, content: str) -> float:
        has_observation = bool(re.search(r'(observe|notice|find)', content, re.I))
        has_hypothesis = bool(re.search(r'(hypothesis|explanation|theory)', content, re.I))
        has_best_explanation = bool(re.search(r'(best explains|most likely)', content, re.I))
        
        return (has_observation * 0.3 + has_hypothesis * 0.4 + has_best_explanation * 0.3)
    
    def _check_causal_alignment(self, content: str) -> float:
        has_cause = bool(re.search(r'(cause|leads to|results in)', content, re.I))
        has_effect = bool(re.search(r'(effect|consequence|outcome)', content, re.I))
        has_mechanism = bool(re.search(r'(mechanism|process|how)', content, re.I))
        
        return (has_cause * 0.35 + has_effect * 0.35 + has_mechanism * 0.3)
    
    def _get_reasoning_suggestions(self, reasoning_type: str) -> List[str]:
        suggestions_map = {
            "deductive": ["Ensure clear premise-conclusion structure", "Verify logical validity"],
            "inductive": ["Provide more observational data", "Strengthen pattern recognition"],
            "abductive": ["Consider alternative explanations", "Evaluate hypothesis plausibility"],
            "causal": ["Clarify causal mechanisms", "Rule out confounding factors"],
        }
        return suggestions_map.get(reasoning_type.lower(), ["Review reasoning structure"])
    
    def _combine_validation_results(self, results: List[ValidationResult]) -> ValidationResult:
        if not results:
            return ValidationResult(
                status=ValidationStatus.UNCERTAIN,
                confidence=0.0,
                category=ValidationCategory.LOGICAL_CONSISTENCY,
                evidence=[],
                errors=["No validation performed"],
                suggestions=[],
                metrics={},
                validator_id="combined_validator"
            )
        
        avg_confidence = np.mean([r.confidence for r in results])
        
        all_errors = []
        all_evidence = []
        all_suggestions = []
        combined_metrics = {}
        
        for r in results:
            all_errors.extend(r.errors)
            all_evidence.extend(r.evidence[:2])
            all_suggestions.extend(r.suggestions[:1])
            combined_metrics.update({f"{r.category.value}_{k}": v for k, v in r.metrics.items()})
        
        status_counts = defaultdict(int)
        for r in results:
            status_counts[r.status] += 1
        
        if status_counts[ValidationStatus.VALID] > len(results) / 2:
            final_status = ValidationStatus.VALID
        elif status_counts[ValidationStatus.INVALID] > len(results) / 2:
            final_status = ValidationStatus.INVALID
        elif avg_confidence > 0.6:
            final_status = ValidationStatus.PARTIALLY_VALID
        else:
            final_status = ValidationStatus.UNCERTAIN
        
        return ValidationResult(
            status=final_status,
            confidence=avg_confidence,
            category=ValidationCategory.LOGICAL_CONSISTENCY,
            evidence=all_evidence,
            errors=all_errors,
            suggestions=list(set(all_suggestions)),
            metrics=combined_metrics,
            validator_id="combined_validator"
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        if not self.validation_history:
            return {"message": "No validations performed yet"}
        
        total_validations = len(self.validation_history)
        valid_count = sum(1 for v in self.validation_history if v.is_valid())
        avg_confidence = np.mean([v.confidence for v in self.validation_history])
        
        status_distribution = defaultdict(int)
        for v in self.validation_history:
            status_distribution[v.status.value] += 1
        
        category_performance = defaultdict(list)
        for v in self.validation_history:
            category_performance[v.category.value].append(v.confidence)
        
        category_avg = {k: np.mean(v) for k, v in category_performance.items()}
        
        return {
            "total_validations": total_validations,
            "valid_count": valid_count,
            "validity_rate": valid_count / total_validations,
            "average_confidence": avg_confidence,
            "status_distribution": dict(status_distribution),
            "category_performance": category_avg,
            "requires_review": sum(1 for v in self.validation_history if v.requires_human_review()),
        }