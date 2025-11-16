import os
import json
import asyncio
from typing import List, Dict
import aiohttp

# Export training data from Supabase to local files
async def fetch_training_data(reasoning_type: str, min_confidence: float = 0.7, limit: int = 10000):
    """Fetch training data from the API"""
    base_url = os.getenv("NEXT_PUBLIC_BASE_URL", "http://localhost:3000")
    
    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/api/training/export"
        params = {
            "type": reasoning_type,
            "minConfidence": min_confidence,
            "limit": limit
        }
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to fetch training data: {response.status}")

def format_training_example(query: str, response: Dict) -> Dict:
    """Format a single training example for fine-tuning"""
    
    # Extract phases from response
    phases = response.get("phases", [])
    
    # Build training text with Crowe Logic Framework structure
    reasoning_text = f"Problem: {query}\n\n"
    
    for phase in phases:
        reasoning_text += f"{phase['name']}:\n{phase['content']}\n\n"
    
    reasoning_text += f"Final Answer: {response.get('answer', '')}"
    
    return {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": reasoning_text}
        ],
        "metadata": {
            "confidence": response.get("overall_confidence", 0),
            "type": response.get("reasoning_type", "unknown")
        }
    }

async def export_training_dataset(
    output_dir: str = "./training_data",
    reasoning_types: List[str] = ["math", "logic", "science", "code"],
    min_confidence: float = 0.7
):
    """Export all training data organized by type"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for reasoning_type in reasoning_types:
        print(f"[v0] Exporting {reasoning_type} training data...")
        
        data = await fetch_training_data(reasoning_type, min_confidence)
        training_examples = []
        
        for item in data.get("data", []):
            example = format_training_example(
                item["query"],
                json.loads(item["response"])
            )
            training_examples.append(example)
        
        # Save to JSONL format (standard for LLM fine-tuning)
        output_file = os.path.join(output_dir, f"{reasoning_type}_train.jsonl")
        with open(output_file, "w") as f:
            for example in training_examples:
                f.write(json.dumps(example) + "\n")
        
        print(f"[v0] Exported {len(training_examples)} examples to {output_file}")
    
    # Create combined dataset
    print("[v0] Creating combined dataset...")
    combined_file = os.path.join(output_dir, "combined_train.jsonl")
    
    with open(combined_file, "w") as outfile:
        for reasoning_type in reasoning_types:
            input_file = os.path.join(output_dir, f"{reasoning_type}_train.jsonl")
            if os.path.exists(input_file):
                with open(input_file, "r") as infile:
                    outfile.write(infile.read())
    
    print(f"[v0] Training data export complete!")

if __name__ == "__main__":
    asyncio.run(export_training_dataset())
