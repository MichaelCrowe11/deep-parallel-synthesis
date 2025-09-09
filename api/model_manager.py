"""
Centralized Model Management for DPS
Supports multiple backends: Ollama, HuggingFace, Local, OpenAI-compatible APIs
"""

import asyncio
import json
import logging
import psutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import ollama
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class ModelBackend(Enum):
    """Supported model backends"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    OPENAI = "openai"
    VLLM = "vllm"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    backend: ModelBackend
    path: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
    """Loaded model information"""
    config: ModelConfig
    model: Any
    tokenizer: Optional[Any] = None
    pipeline: Optional[Any] = None
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    memory_usage: float = 0.0
    request_count: int = 0


class ModelManager:
    """
    Centralized model management system
    Handles loading, unloading, and inference across multiple backends
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.models_config: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.default_backend = ModelBackend.OLLAMA
        self.cache_dir = Path("./models_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        else:
            self._load_default_config()
    
    def _load_config(self, config_path: str):
        """Load model configurations from file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        for model_name, model_config in config_data.items():
            backend = ModelBackend(model_config["backend"])
            self.models_config[model_name] = ModelConfig(
                name=model_name,
                backend=backend,
                path=model_config.get("path"),
                api_url=model_config.get("api_url"),
                api_key=model_config.get("api_key"),
                parameters=model_config.get("parameters", {}),
                hardware_requirements=model_config.get("hardware_requirements", {}),
                metadata=model_config.get("metadata", {})
            )
    
    def _load_default_config(self):
        """Load default model configurations"""
        self.models_config = {
            "llama3.1:70b": ModelConfig(
                name="llama3.1:70b",
                backend=ModelBackend.OLLAMA,
                parameters={"temperature": 0.7, "top_p": 0.9},
                hardware_requirements={"min_ram": 140, "min_vram": 80}
            ),
            "llama3.1:8b": ModelConfig(
                name="llama3.1:8b",
                backend=ModelBackend.OLLAMA,
                parameters={"temperature": 0.7, "top_p": 0.9},
                hardware_requirements={"min_ram": 16, "min_vram": 8}
            ),
            "deepparallel": ModelConfig(
                name="Mcrowe1210/DeepParallel",
                backend=ModelBackend.OLLAMA,
                parameters={"temperature": 0.7},
                hardware_requirements={"min_ram": 32, "min_vram": 16}
            ),
            "gpt2": ModelConfig(
                name="gpt2",
                backend=ModelBackend.HUGGINGFACE,
                path="gpt2",
                parameters={"max_length": 1024},
                hardware_requirements={"min_ram": 4, "min_vram": 2}
            ),
            "mistral-7b": ModelConfig(
                name="mistralai/Mistral-7B-v0.1",
                backend=ModelBackend.HUGGINGFACE,
                path="mistralai/Mistral-7B-v0.1",
                parameters={"temperature": 0.7, "max_new_tokens": 512},
                hardware_requirements={"min_ram": 16, "min_vram": 14}
            )
        }
    
    async def initialize(self):
        """Initialize model manager"""
        logger.info("Initializing Model Manager...")
        
        # Check available hardware
        self.hardware_info = self._get_hardware_info()
        logger.info(f"Hardware: {self.hardware_info}")
        
        # Check backend availability
        await self._check_backends()
        
        # Load default model if possible
        await self._load_default_model()
        
        logger.info("Model Manager initialized")
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    async def _check_backends(self):
        """Check which backends are available"""
        self.available_backends = {}
        
        # Check Ollama
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    self.available_backends[ModelBackend.OLLAMA] = True
                    logger.info("Ollama backend available")
        except:
            self.available_backends[ModelBackend.OLLAMA] = False
            logger.warning("Ollama backend not available")
        
        # Check HuggingFace
        try:
            from transformers import AutoModel
            self.available_backends[ModelBackend.HUGGINGFACE] = True
            logger.info("HuggingFace backend available")
        except:
            self.available_backends[ModelBackend.HUGGINGFACE] = False
            logger.warning("HuggingFace backend not available")
        
        # Local is always available
        self.available_backends[ModelBackend.LOCAL] = True
    
    async def _load_default_model(self):
        """Load a default model based on available resources"""
        # Try to load smallest model that fits in memory
        for model_name, config in self.models_config.items():
            if not self.available_backends.get(config.backend, False):
                continue
            
            req_ram = config.hardware_requirements.get("min_ram", 0)
            if req_ram <= self.hardware_info["available_ram_gb"]:
                try:
                    await self.load_model(model_name)
                    logger.info(f"Loaded default model: {model_name}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
        
        logger.warning("No default model could be loaded")
    
    async def load_model(self, model_name: str, backend: Optional[str] = None) -> LoadedModel:
        """Load a model into memory"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        # Get config
        if model_name in self.models_config:
            config = self.models_config[model_name]
        else:
            # Create new config
            config = ModelConfig(
                name=model_name,
                backend=ModelBackend(backend) if backend else self.default_backend
            )
        
        # Check hardware requirements
        if not self._check_hardware_requirements(config):
            raise ValueError(f"Insufficient hardware for model {model_name}")
        
        # Load based on backend
        loaded_model = await self._load_model_by_backend(config)
        
        # Store loaded model
        self.loaded_models[model_name] = loaded_model
        
        # Update memory usage
        loaded_model.memory_usage = self._estimate_memory_usage(loaded_model)
        
        logger.info(f"Loaded model {model_name} ({config.backend.value})")
        return loaded_model
    
    def _check_hardware_requirements(self, config: ModelConfig) -> bool:
        """Check if hardware meets model requirements"""
        req_ram = config.hardware_requirements.get("min_ram", 0)
        req_vram = config.hardware_requirements.get("min_vram", 0)
        
        if req_ram > self.hardware_info["available_ram_gb"]:
            logger.warning(f"Insufficient RAM: required {req_ram}GB, available {self.hardware_info['available_ram_gb']}GB")
            return False
        
        if req_vram > 0 and self.hardware_info.get("vram_gb", 0) < req_vram:
            logger.warning(f"Insufficient VRAM: required {req_vram}GB, available {self.hardware_info.get('vram_gb', 0)}GB")
            return False
        
        return True
    
    async def _load_model_by_backend(self, config: ModelConfig) -> LoadedModel:
        """Load model based on backend type"""
        if config.backend == ModelBackend.OLLAMA:
            return await self._load_ollama_model(config)
        elif config.backend == ModelBackend.HUGGINGFACE:
            return await self._load_huggingface_model(config)
        elif config.backend == ModelBackend.LOCAL:
            return await self._load_local_model(config)
        elif config.backend == ModelBackend.OPENAI:
            return await self._load_openai_model(config)
        elif config.backend == ModelBackend.VLLM:
            return await self._load_vllm_model(config)
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")
    
    async def _load_ollama_model(self, config: ModelConfig) -> LoadedModel:
        """Load Ollama model"""
        # Pull model if not available
        try:
            ollama.pull(config.name)
        except:
            logger.info(f"Model {config.name} already available in Ollama")
        
        # Create client
        client = ollama.Client()
        
        return LoadedModel(
            config=config,
            model=client,
            tokenizer=None,
            pipeline=None
        )
    
    async def _load_huggingface_model(self, config: ModelConfig) -> LoadedModel:
        """Load HuggingFace model"""
        model_path = config.path or config.name
        
        # Download if needed
        local_path = self.cache_dir / model_path.replace("/", "_")
        if not local_path.exists():
            logger.info(f"Downloading {model_path} from HuggingFace...")
            snapshot_download(model_path, local_dir=local_path)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **config.parameters
        )
        
        return LoadedModel(
            config=config,
            model=model,
            tokenizer=tokenizer,
            pipeline=pipe
        )
    
    async def _load_local_model(self, config: ModelConfig) -> LoadedModel:
        """Load local model from disk"""
        if not config.path:
            raise ValueError("Local model requires path")
        
        model_path = Path(config.path)
        if not model_path.exists():
            raise ValueError(f"Model path {model_path} does not exist")
        
        # Load based on file extension
        if model_path.suffix in [".pt", ".pth"]:
            model = torch.load(model_path)
        elif model_path.suffix == ".onnx":
            import onnxruntime as ort
            model = ort.InferenceSession(str(model_path))
        else:
            # Assume it's a HuggingFace model directory
            return await self._load_huggingface_model(config)
        
        return LoadedModel(
            config=config,
            model=model,
            tokenizer=None,
            pipeline=None
        )
    
    async def _load_openai_model(self, config: ModelConfig) -> LoadedModel:
        """Load OpenAI-compatible API model"""
        if not config.api_url:
            config.api_url = "https://api.openai.com/v1"
        
        # Create client
        client = httpx.AsyncClient(
            base_url=config.api_url,
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        )
        
        return LoadedModel(
            config=config,
            model=client,
            tokenizer=None,
            pipeline=None
        )
    
    async def _load_vllm_model(self, config: ModelConfig) -> LoadedModel:
        """Load vLLM model"""
        if not config.api_url:
            config.api_url = "http://localhost:8000"
        
        # Create client for vLLM server
        client = httpx.AsyncClient(base_url=config.api_url)
        
        return LoadedModel(
            config=config,
            model=client,
            tokenizer=None,
            pipeline=None
        )
    
    def _estimate_memory_usage(self, loaded_model: LoadedModel) -> float:
        """Estimate memory usage of loaded model"""
        if loaded_model.config.backend == ModelBackend.HUGGINGFACE and loaded_model.model:
            # Estimate based on parameter count
            param_count = sum(p.numel() for p in loaded_model.model.parameters())
            # Rough estimate: 4 bytes per parameter
            return (param_count * 4) / (1024**3)  # Convert to GB
        
        # Return from config if available
        return loaded_model.config.hardware_requirements.get("min_ram", 0)
    
    async def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        loaded_model = self.loaded_models[model_name]
        
        # Cleanup based on backend
        if loaded_model.config.backend == ModelBackend.HUGGINGFACE:
            del loaded_model.model
            del loaded_model.tokenizer
            del loaded_model.pipeline
            torch.cuda.empty_cache()
        elif loaded_model.config.backend in [ModelBackend.OPENAI, ModelBackend.VLLM]:
            await loaded_model.model.aclose()
        
        del self.loaded_models[model_name]
        logger.info(f"Unloaded model {model_name}")
    
    async def get_model(self, model_name: str) -> LoadedModel:
        """Get a loaded model, loading it if necessary"""
        if model_name not in self.loaded_models:
            await self.load_model(model_name)
        
        loaded_model = self.loaded_models[model_name]
        loaded_model.last_used = time.time()
        loaded_model.request_count += 1
        
        return loaded_model
    
    async def infer(
        self,
        model_name: str,
        prompt: str,
        **kwargs
    ) -> str:
        """Run inference on a model"""
        loaded_model = await self.get_model(model_name)
        
        if loaded_model.config.backend == ModelBackend.OLLAMA:
            response = loaded_model.model.generate(
                model=loaded_model.config.name,
                prompt=prompt,
                **kwargs
            )
            return response["response"]
        
        elif loaded_model.config.backend == ModelBackend.HUGGINGFACE:
            result = loaded_model.pipeline(
                prompt,
                **kwargs
            )
            return result[0]["generated_text"]
        
        elif loaded_model.config.backend == ModelBackend.OPENAI:
            response = await loaded_model.model.post(
                "/chat/completions",
                json={
                    "model": loaded_model.config.name,
                    "messages": [{"role": "user", "content": prompt}],
                    **kwargs
                }
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        
        elif loaded_model.config.backend == ModelBackend.VLLM:
            response = await loaded_model.model.post(
                "/v1/completions",
                json={
                    "model": loaded_model.config.name,
                    "prompt": prompt,
                    **kwargs
                }
            )
            data = response.json()
            return data["choices"][0]["text"]
        
        else:
            raise ValueError(f"Inference not supported for backend {loaded_model.config.backend}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models = []
        
        # Add configured models
        for name, config in self.models_config.items():
            model_info = {
                "name": name,
                "backend": config.backend.value,
                "loaded": name in self.loaded_models,
                "available": self.available_backends.get(config.backend, False)
            }
            
            if name in self.loaded_models:
                loaded = self.loaded_models[name]
                model_info.update({
                    "memory_usage": loaded.memory_usage,
                    "last_used": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(loaded.last_used)),
                    "request_count": loaded.request_count
                })
            
            models.append(model_info)
        
        # Add Ollama models if available
        if self.available_backends.get(ModelBackend.OLLAMA):
            try:
                ollama_models = ollama.list()
                for model in ollama_models.get("models", []):
                    if model["name"] not in self.models_config:
                        models.append({
                            "name": model["name"],
                            "backend": "ollama",
                            "loaded": model["name"] in self.loaded_models,
                            "available": True,
                            "size_gb": model.get("size", 0) / (1024**3)
                        })
            except:
                pass
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if model_name in self.loaded_models:
            loaded = self.loaded_models[model_name]
            return {
                "name": model_name,
                "backend": loaded.config.backend.value,
                "loaded": True,
                "memory_usage": loaded.memory_usage,
                "last_used": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(loaded.last_used)),
                "request_count": loaded.request_count,
                "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(loaded.loaded_at)),
                "parameters": loaded.config.parameters,
                "hardware_requirements": loaded.config.hardware_requirements
            }
        elif model_name in self.models_config:
            config = self.models_config[model_name]
            return {
                "name": model_name,
                "backend": config.backend.value,
                "loaded": False,
                "parameters": config.parameters,
                "hardware_requirements": config.hardware_requirements
            }
        else:
            raise ValueError(f"Model {model_name} not found")
    
    async def cleanup_unused(self, max_idle_minutes: int = 30):
        """Cleanup models that haven't been used recently"""
        current_time = time.time()
        max_idle_seconds = max_idle_minutes * 60
        
        models_to_unload = []
        for name, loaded in self.loaded_models.items():
            if current_time - loaded.last_used > max_idle_seconds:
                models_to_unload.append(name)
        
        for name in models_to_unload:
            logger.info(f"Unloading idle model: {name}")
            await self.unload_model(name)
        
        return len(models_to_unload)