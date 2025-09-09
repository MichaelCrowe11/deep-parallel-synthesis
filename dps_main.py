#!/usr/bin/env python3
"""
Deep Parallel Synthesis - Main Entry Point
Unified interface for all DPS functionality
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

# Import core modules
from dps_core import DPSCore, ReasoningType
from dps.model import DPSModel, DPSConfig
from dps.reasoning import ParallelReasoningChains
from dps.validator import ScientificValidator
from dps.training import DPSTrainer, ScientificReasoningDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPSRequest(BaseModel):
    """Request model for DPS API"""
    prompt: str = Field(..., description="The input prompt or question")
    reasoning_types: Optional[list] = Field(None, description="Specific reasoning types to use")
    max_depth: Optional[int] = Field(5, description="Maximum reasoning depth")
    num_chains: Optional[int] = Field(8, description="Number of parallel chains")
    temperature: Optional[float] = Field(0.7, description="Synthesis temperature")
    validate: Optional[bool] = Field(True, description="Whether to validate output")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class DPSResponse(BaseModel):
    """Response model for DPS API"""
    response: str
    confidence: float
    reasoning_chains: list
    metrics: Dict[str, Any]
    evidence: list


class DPSSystem:
    """Main DPS System orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.core = None
        self.model = None
        self.validator = ScientificValidator()
        self.app = self._create_app()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "num_chains": 8,
            "max_depth": 5,
            "synthesis_temperature": 0.7,
            "validation_threshold": 0.85,
            "enable_quantum": False,
            "enable_neural_evolution": False,
            "model_path": None,
            "use_ollama": True,
            "ollama_model": "llama3.1:70b",
            "api_port": 8000,
            "api_host": "0.0.0.0"
        }
    
    def initialize(self):
        """Initialize the DPS system"""
        logger.info("Initializing DPS System...")
        
        # Initialize core
        self.core = DPSCore(
            num_chains=self.config["num_chains"],
            max_depth=self.config["max_depth"],
            synthesis_temperature=self.config["synthesis_temperature"],
            validation_threshold=self.config["validation_threshold"],
            enable_quantum=self.config["enable_quantum"],
            enable_neural_evolution=self.config["enable_neural_evolution"]
        )
        
        # Load model if specified
        if self.config.get("model_path"):
            self._load_model(self.config["model_path"])
        
        logger.info("DPS System initialized successfully")
    
    def _load_model(self, model_path: str):
        """Load a trained DPS model"""
        try:
            config_path = Path(model_path) / "dps_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    model_config = DPSConfig(**json.load(f))
                self.model = DPSModel(model_config)
                logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Deep Parallel Synthesis API",
            version="1.0.0",
            description="Advanced scientific reasoning through parallel synthesis"
        )
        
        # Enable CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define routes
        @app.get("/")
        async def root():
            return {
                "service": "Deep Parallel Synthesis",
                "version": "1.0.0",
                "status": "running"
            }
        
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "core_initialized": self.core is not None,
                "model_loaded": self.model is not None
            }
        
        @app.post("/reason", response_model=DPSResponse)
        async def reason(request: DPSRequest):
            """Main reasoning endpoint"""
            if not self.core:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            try:
                # Convert reasoning types from strings to enums
                reasoning_types = None
                if request.reasoning_types:
                    reasoning_types = [
                        ReasoningType[rt.upper()] 
                        for rt in request.reasoning_types
                    ]
                
                # Update core parameters if specified
                if request.num_chains:
                    self.core.num_chains = request.num_chains
                if request.max_depth:
                    self.core.max_depth = request.max_depth
                if request.temperature:
                    self.core.synthesis_temperature = request.temperature
                
                # Execute reasoning
                result = await self.core.reason(
                    prompt=request.prompt,
                    context=request.context,
                    reasoning_types=reasoning_types
                )
                
                # Validate if requested
                if request.validate:
                    validation = self.validator.validate(
                        content=result["response"],
                        reasoning_type="systematic",
                        evidence=result.get("evidence", [])
                    )
                    result["validation"] = validation.to_dict()
                
                return DPSResponse(
                    response=result["response"],
                    confidence=result["confidence"],
                    reasoning_chains=result["reasoning_chains"],
                    metrics=result["metrics"],
                    evidence=result.get("evidence", [])
                )
                
            except Exception as e:
                logger.error(f"Reasoning error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics")
        async def get_metrics():
            """Get system metrics"""
            if not self.core:
                return {"error": "System not initialized"}
            return self.core.get_metrics()
        
        @app.post("/clear")
        async def clear_history():
            """Clear reasoning history"""
            if self.core:
                self.core.clear_history()
            return {"status": "cleared"}
        
        @app.get("/config")
        async def get_config():
            """Get current configuration"""
            return self.config
        
        @app.post("/config")
        async def update_config(config: Dict[str, Any]):
            """Update configuration"""
            self.config.update(config)
            return {"status": "updated", "config": self.config}
        
        return app
    
    def run_api(self):
        """Run the API server"""
        if not self.core:
            self.initialize()
        
        uvicorn.run(
            self.app,
            host=self.config["api_host"],
            port=self.config["api_port"],
            log_level="info"
        )
    
    async def run_cli(self, prompt: str, **kwargs):
        """Run in CLI mode"""
        if not self.core:
            self.initialize()
        
        result = await self.core.reason(prompt, **kwargs)
        return result
    
    def train(self, train_data: str, eval_data: Optional[str] = None, **kwargs):
        """Train a new DPS model"""
        logger.info("Starting training...")
        
        # Create config
        model_config = DPSConfig(
            base_model_name=kwargs.get("base_model", "meta-llama/Llama-3.1-70B"),
            num_parallel_chains=kwargs.get("num_chains", 8),
            reasoning_depth=kwargs.get("max_depth", 5)
        )
        
        # Create trainer
        trainer = DPSTrainer(
            model_config=model_config,
            training_args=kwargs,
            output_dir=kwargs.get("output_dir", "./dps_output")
        )
        
        # Load datasets
        tokenizer = trainer.tokenizer
        train_dataset = ScientificReasoningDataset(
            train_data,
            tokenizer,
            max_length=kwargs.get("max_length", 2048)
        )
        
        eval_dataset = None
        if eval_data:
            eval_dataset = ScientificReasoningDataset(
                eval_data,
                tokenizer,
                max_length=kwargs.get("max_length", 2048)
            )
        
        # Train
        trainer.train(
            train_dataset,
            eval_dataset,
            num_epochs=kwargs.get("num_epochs", 3)
        )
        
        logger.info("Training completed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deep Parallel Synthesis System")
    parser.add_argument("command", choices=["api", "cli", "train"], help="Command to run")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--prompt", type=str, help="Prompt for CLI mode")
    parser.add_argument("--train-data", type=str, help="Training data path")
    parser.add_argument("--eval-data", type=str, help="Evaluation data path")
    parser.add_argument("--output-dir", type=str, default="./dps_output", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    
    args = parser.parse_args()
    
    # Create system
    system = DPSSystem(config_path=args.config)
    
    if args.command == "api":
        # Override config with command line args
        if args.port:
            system.config["api_port"] = args.port
        if args.host:
            system.config["api_host"] = args.host
        
        # Run API server
        system.run_api()
    
    elif args.command == "cli":
        if not args.prompt:
            print("Error: --prompt is required for CLI mode")
            sys.exit(1)
        
        # Run in CLI mode
        async def run():
            result = await system.run_cli(args.prompt)
            print("\n" + "="*80)
            print("RESPONSE:")
            print("="*80)
            print(result["response"])
            print("\n" + "="*80)
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Chains used: {len(result['reasoning_chains'])}")
            print(f"Processing time: {result['metrics'].get('total_time', 0):.2f}s")
        
        asyncio.run(run())
    
    elif args.command == "train":
        if not args.train_data:
            print("Error: --train-data is required for training")
            sys.exit(1)
        
        # Run training
        system.train(
            train_data=args.train_data,
            eval_data=args.eval_data,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )


if __name__ == "__main__":
    main()