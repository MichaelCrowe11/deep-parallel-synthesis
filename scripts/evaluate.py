#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from eval.metrics import DPSEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPS Model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the DPS model to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset (JSON format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "logical_consistency", "reasoning_depth", "convergence_quality", "scientific_accuracy"],
        help="Metrics to evaluate"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate detailed evaluation report"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    
    print("=== DPS Model Evaluation ===")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metrics: {args.metrics}")
    print()
    
    evaluator = DPSEvaluator(model_path=args.model_path)
    
    print("Running evaluation...")
    results = evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        output_dir=args.output_dir
    )
    
    print("\n=== Evaluation Results ===")
    for metric_name, result in results.items():
        if hasattr(result, 'score'):
            print(f"{metric_name}: {result.score:.4f}")
        elif isinstance(result, dict) and 'accuracy' in result:
            print(f"{metric_name} accuracy: {result['accuracy']:.4f}")
    
    if args.generate_report:
        print("\nGenerating detailed report...")
        report = evaluator.generate_report(output_dir=args.output_dir)
        print(f"Report saved to {args.output_dir}/evaluation_report.md")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()