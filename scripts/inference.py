#!/usr/bin/env python3

import argparse
import json
import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from serving.inference_server import DPSInferenceEngine, InferenceRequest


async def interactive_mode(engine: DPSInferenceEngine):
    print("=== DPS Interactive Inference Mode ===")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for options")
    print()
    
    while True:
        try:
            prompt = input("Query> ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("Exiting...")
                break
            
            if prompt.lower() == 'help':
                print("\nOptions:")
                print("  /synthesis on|off  - Toggle parallel synthesis")
                print("  /validate on|off   - Toggle output validation")
                print("  /depth N          - Set reasoning depth (1-10)")
                print("  /chains N         - Set number of parallel chains (1-16)")
                print("  /temp F           - Set temperature (0.0-2.0)")
                print("  /max_tokens N     - Set max tokens (1-4096)")
                print()
                continue
            
            if prompt.startswith('/'):
                parts = prompt[1:].split()
                if len(parts) >= 2:
                    command, value = parts[0], parts[1]
                    if command == 'synthesis':
                        engine.use_synthesis = value.lower() == 'on'
                        print(f"Parallel synthesis: {engine.use_synthesis}")
                    elif command == 'validate':
                        engine.validate = value.lower() == 'on'
                        print(f"Output validation: {engine.validate}")
                    elif command == 'depth':
                        engine.reasoning_chains.max_depth = int(value)
                        print(f"Reasoning depth: {engine.reasoning_chains.max_depth}")
                    elif command == 'chains':
                        engine.reasoning_chains.num_chains = int(value)
                        print(f"Parallel chains: {engine.reasoning_chains.num_chains}")
                    elif command == 'temp':
                        engine.temperature = float(value)
                        print(f"Temperature: {engine.temperature}")
                    elif command == 'max_tokens':
                        engine.max_tokens = int(value)
                        print(f"Max tokens: {engine.max_tokens}")
                continue
            
            if not prompt:
                continue
            
            request = InferenceRequest(
                prompt=prompt,
                max_tokens=getattr(engine, 'max_tokens', 512),
                temperature=getattr(engine, 'temperature', 0.7),
                use_parallel_synthesis=getattr(engine, 'use_synthesis', True),
                validate_output=getattr(engine, 'validate', True)
            )
            
            print("\nGenerating response...")
            response = await engine.generate(request)
            
            print(f"\n{'='*50}")
            print("Response:")
            print(response.generated_text)
            print(f"{'='*50}")
            
            if response.validation_status:
                print(f"Validation: {response.validation_status}")
            print(f"Confidence: {response.confidence_score:.4f}")
            print(f"Generation time: {response.generation_time:.2f}s")
            
            if response.synthesis_metrics:
                print(f"Synthesis quality: {response.synthesis_metrics['synthesis_quality']:.4f}")
                print(f"Total nodes explored: {response.synthesis_metrics['num_nodes']}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


async def batch_mode(engine: DPSInferenceEngine, input_file: str, output_file: str):
    print(f"Processing batch from {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    prompts = []
    if isinstance(data, list):
        prompts = [item.get("prompt", item) if isinstance(item, dict) else item for item in data]
    elif isinstance(data, dict) and "prompts" in data:
        prompts = data["prompts"]
    else:
        raise ValueError("Invalid input format. Expected list or dict with 'prompts' key")
    
    print(f"Processing {len(prompts)} prompts...")
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}")
        
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=512,
            use_parallel_synthesis=True,
            validate_output=True
        )
        
        response = await engine.generate(request)
        
        results.append({
            "prompt": prompt,
            "response": response.generated_text,
            "confidence": response.confidence_score,
            "validation": response.validation_status,
            "generation_time": response.generation_time,
            "synthesis_metrics": response.synthesis_metrics
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


async def single_inference(engine: DPSInferenceEngine, prompt: str):
    request = InferenceRequest(
        prompt=prompt,
        max_tokens=512,
        temperature=0.7,
        use_parallel_synthesis=True,
        validate_output=True
    )
    
    response = await engine.generate(request)
    
    print("\n" + "="*50)
    print("Response:")
    print(response.generated_text)
    print("="*50)
    print(f"Confidence: {response.confidence_score:.4f}")
    if response.validation_status:
        print(f"Validation: {response.validation_status}")
    print(f"Generation time: {response.generation_time:.2f}s")
    
    if response.synthesis_metrics:
        print("\nSynthesis Metrics:")
        for key, value in response.synthesis_metrics.items():
            print(f"  {key}: {value}")


async def main():
    parser = argparse.ArgumentParser(description="DPS Inference Script")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the DPS model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "single", "batch"],
        default="interactive",
        help="Inference mode"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for inference (single mode)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file for batch inference"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="inference_results.json",
        help="Output file for batch inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU"
    )
    
    args = parser.parse_args()
    
    print("Initializing DPS Inference Engine...")
    engine = DPSInferenceEngine(
        model_path=args.model_path,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    if args.mode == "interactive":
        await interactive_mode(engine)
    elif args.mode == "single":
        if not args.prompt:
            print("Error: --prompt required for single mode")
            return
        await single_inference(engine, args.prompt)
    elif args.mode == "batch":
        if not args.input_file:
            print("Error: --input-file required for batch mode")
            return
        await batch_mode(engine, args.input_file, args.output_file)


if __name__ == "__main__":
    asyncio.run(main())