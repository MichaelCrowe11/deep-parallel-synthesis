import asyncio
import torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import json
from pathlib import Path
import logging
from contextlib import asynccontextmanager

import sys
sys.path.append(str(Path(__file__).parent.parent))

from dps.model import DPSModel, DPSConfig
from dps.reasoning import ParallelReasoningChains, ReasoningNode
from dps.validator import ScientificValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=100)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0)
    use_parallel_synthesis: bool = Field(default=True)
    num_parallel_chains: int = Field(default=8, ge=1, le=16)
    reasoning_depth: int = Field(default=5, ge=1, le=10)
    validate_output: bool = Field(default=True)
    stream: bool = Field(default=False)


class InferenceResponse(BaseModel):
    generated_text: str
    confidence_score: float
    validation_status: Optional[str] = None
    reasoning_paths: Optional[List[Dict[str, Any]]] = None
    synthesis_metrics: Optional[Dict[str, Any]] = None
    generation_time: float


class BatchInferenceRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    use_parallel_synthesis: bool = Field(default=True)


class DPSInferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        max_batch_size: int = 32,
        use_flash_attention: bool = True
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.max_batch_size = max_batch_size
        
        with open(self.model_path / "dps_config.json", "r") as f:
            config_dict = json.load(f)
        self.config = DPSConfig(**config_dict)
        
        self.vllm_engine = LLM(
            model=str(self.model_path),
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=8192,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            enforce_eager=not use_flash_attention,
        )
        
        self.validator = ScientificValidator()
        self.reasoning_chains = ParallelReasoningChains(
            num_chains=self.config.num_parallel_chains,
            max_depth=self.config.reasoning_depth
        )
        
        logger.info(f"DPS Inference Engine initialized with model from {model_path}")
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        import time
        start_time = time.time()
        
        if request.use_parallel_synthesis:
            response = await self._generate_with_synthesis(request)
        else:
            response = await self._generate_standard(request)
        
        generation_time = time.time() - start_time
        response.generation_time = generation_time
        
        return response
    
    async def _generate_standard(self, request: InferenceRequest) -> InferenceResponse:
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        
        outputs = self.vllm_engine.generate(
            [request.prompt],
            sampling_params
        )
        
        generated_text = outputs[0].outputs[0].text
        
        validation_result = None
        if request.validate_output:
            validation_result = self.validator.validate(
                content=generated_text,
                reasoning_type="systematic"
            )
        
        return InferenceResponse(
            generated_text=generated_text,
            confidence_score=validation_result.confidence if validation_result else 0.0,
            validation_status=validation_result.status.value if validation_result else None,
            reasoning_paths=None,
            synthesis_metrics=None,
            generation_time=0.0
        )
    
    async def _generate_with_synthesis(self, request: InferenceRequest) -> InferenceResponse:
        self.reasoning_chains.initialize_chains(request.prompt)
        
        async def expand_function(node, chain_id):
            expansion_prompt = f"{request.prompt}\n\nContinuing from: {node.content}\n\nNext reasoning step:"
            
            sampling_params = SamplingParams(
                max_tokens=128,
                temperature=request.temperature * 1.2,
                top_p=request.top_p,
            )
            
            outputs = self.vllm_engine.generate(
                [expansion_prompt],
                sampling_params
            )
            
            generated = outputs[0].outputs[0].text
            
            expansions = []
            for i in range(3):
                expansions.append({
                    "content": f"{generated} [Variation {i+1}]",
                    "type": "DEDUCTIVE" if i == 0 else "INDUCTIVE" if i == 1 else "CAUSAL",
                    "confidence": 0.8 - i * 0.1,
                    "supporting": [],
                    "contradicting": []
                })
            
            return expansions
        
        for depth in range(request.reasoning_depth):
            await self.reasoning_chains.parallel_expand(expand_function, depth)
            self.reasoning_chains.cross_pollinate_chains()
            
            if depth > 2:
                for chain in self.reasoning_chains.chains:
                    chain.prune_low_confidence_paths(threshold=0.3)
        
        synthesis = self.reasoning_chains.synthesize_chains()
        
        best_path = self.reasoning_chains.get_best_reasoning_path()
        if best_path:
            synthesis_prompt = f"{request.prompt}\n\nBased on the reasoning path:\n{' -> '.join(best_path[:3])}\n\nFinal answer:"
        else:
            synthesis_prompt = request.prompt
        
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        
        final_outputs = self.vllm_engine.generate(
            [synthesis_prompt],
            sampling_params
        )
        
        generated_text = final_outputs[0].outputs[0].text
        
        validation_result = None
        if request.validate_output:
            validation_result = self.validator.validate(
                content=generated_text,
                reasoning_type="systematic",
                evidence=[p["path"][0] for p in synthesis.get("top_reasoning_paths", [])]
            )
        
        return InferenceResponse(
            generated_text=generated_text,
            confidence_score=synthesis.get("synthesis_quality", 0.0),
            validation_status=validation_result.status.value if validation_result else None,
            reasoning_paths=synthesis.get("top_reasoning_paths", []),
            synthesis_metrics={
                "num_nodes": synthesis.get("num_total_nodes", 0),
                "convergence_strength": synthesis.get("convergence_strength", 0.0),
                "synthesis_quality": synthesis.get("synthesis_quality", 0.0),
            },
            generation_time=0.0
        )
    
    async def generate_batch(self, request: BatchInferenceRequest) -> List[InferenceResponse]:
        tasks = []
        for prompt in request.prompts:
            inference_req = InferenceRequest(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                use_parallel_synthesis=request.use_parallel_synthesis
            )
            tasks.append(self.generate(inference_req))
        
        responses = await asyncio.gather(*tasks)
        return responses


class DPSServer:
    def __init__(self, engine: DPSInferenceEngine):
        self.engine = engine
        self.app = FastAPI(
            title="Deep Parallel Synthesis Inference Server",
            version="0.0.1",
            description="Advanced scientific reasoning model inference API"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            return {
                "name": "DPS Inference Server",
                "version": "0.0.1",
                "status": "running",
                "model_config": self.engine.config.to_dict()
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @self.app.post("/v1/generate", response_model=InferenceResponse)
        async def generate(request: InferenceRequest):
            try:
                response = await self.engine.generate(request)
                return response
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/generate_batch", response_model=List[InferenceResponse])
        async def generate_batch(request: BatchInferenceRequest):
            try:
                if len(request.prompts) > self.engine.max_batch_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Batch size {len(request.prompts)} exceeds maximum {self.engine.max_batch_size}"
                    )
                
                responses = await self.engine.generate_batch(request)
                return responses
            except Exception as e:
                logger.error(f"Batch generation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/validate")
        async def validate(content: str, reasoning_type: str = "systematic"):
            try:
                validation_result = self.engine.validator.validate(
                    content=content,
                    reasoning_type=reasoning_type
                )
                return validation_result.to_dict()
            except Exception as e:
                logger.error(f"Validation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/v1/model_info")
        async def model_info():
            return {
                "model_path": str(self.engine.model_path),
                "config": self.engine.config.to_dict(),
                "device": self.engine.device,
                "tensor_parallel_size": self.engine.tensor_parallel_size,
                "max_batch_size": self.engine.max_batch_size
            }
        
        @self.app.websocket("/v1/stream")
        async def stream_generate(websocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_json()
                    request = InferenceRequest(**data)
                    
                    if not request.stream:
                        response = await self.engine.generate(request)
                        await websocket.send_json(response.dict())
                    else:
                        await websocket.send_json({
                            "status": "generating",
                            "message": "Streaming generation started"
                        })
                        
                        response = await self.engine.generate(request)
                        
                        chunks = response.generated_text.split()
                        for i, chunk in enumerate(chunks):
                            await websocket.send_json({
                                "status": "streaming",
                                "chunk": chunk + " ",
                                "progress": (i + 1) / len(chunks)
                            })
                            await asyncio.sleep(0.05)
                        
                        await websocket.send_json({
                            "status": "complete",
                            "response": response.dict()
                        })
                        
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({
                    "status": "error",
                    "error": str(e)
                })
            finally:
                await websocket.close()
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="DPS Inference Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the DPS model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    engine = DPSInferenceEngine(
        model_path=args.model_path,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        max_batch_size=args.max_batch_size
    )
    
    server = DPSServer(engine)
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    asyncio.run(main())