#!/usr/bin/env python3
"""
Deep Parallel Synthesis API Server
Production-ready API with OpenAPI documentation, authentication, and monitoring
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security, status, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import jwt
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis
from motor.motor_asyncio import AsyncIOMotorClient
import httpx

# Import DPS modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from dps_core import DPSCore, ReasoningType
from dps.validator import ScientificValidator
from api.model_manager import ModelManager
from api.auth_manager import AuthManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
request_counter = Counter('dps_api_requests_total', 'Total API requests', ['endpoint', 'method'])
request_duration = Histogram('dps_api_request_duration_seconds', 'API request duration', ['endpoint'])
active_connections = Gauge('dps_api_active_connections', 'Active connections')
reasoning_tasks = Counter('dps_reasoning_tasks_total', 'Total reasoning tasks', ['status'])
model_load_time = Histogram('dps_model_load_time_seconds', 'Model loading time')


# Request/Response Models
class ReasoningRequest(BaseModel):
    """Request model for reasoning endpoint"""
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt")
    reasoning_types: Optional[List[str]] = Field(None, description="Reasoning types to use")
    max_depth: int = Field(5, ge=1, le=10, description="Maximum reasoning depth")
    num_chains: int = Field(8, ge=1, le=16, description="Number of parallel chains")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Synthesis temperature")
    validate: bool = Field(True, description="Validate output")
    stream: bool = Field(False, description="Stream response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    model_backend: Optional[str] = Field("ollama", description="Model backend to use")
    
    @validator('reasoning_types')
    def validate_reasoning_types(cls, v):
        if v:
            valid_types = [t.value for t in ReasoningType]
            for rt in v:
                if rt.lower() not in valid_types:
                    raise ValueError(f"Invalid reasoning type: {rt}")
        return v


class ReasoningResponse(BaseModel):
    """Response model for reasoning endpoint"""
    id: str = Field(..., description="Unique task ID")
    response: str = Field(..., description="Generated response")
    confidence: float = Field(..., description="Confidence score")
    reasoning_chains: List[Dict] = Field(..., description="Reasoning chain details")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    evidence: List[str] = Field(..., description="Supporting evidence")
    validation: Optional[Dict] = Field(None, description="Validation results")
    timestamp: str = Field(..., description="Response timestamp")


class BatchRequest(BaseModel):
    """Request model for batch processing"""
    prompts: List[str] = Field(..., min_items=1, max_items=100)
    common_params: Optional[ReasoningRequest] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    uptime: float
    models_loaded: int
    active_tasks: int
    redis_connected: bool
    mongodb_connected: bool


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    backend: str
    loaded: bool
    memory_usage: Optional[float]
    last_used: Optional[str]


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting DPS API Server...")
    
    # Initialize components
    app.state.start_time = time.time()
    app.state.dps_core = DPSCore(
        num_chains=8,
        max_depth=5,
        synthesis_temperature=0.7,
        validation_threshold=0.85
    )
    app.state.validator = ScientificValidator()
    app.state.model_manager = ModelManager()
    app.state.auth_manager = AuthManager()
    
    # Initialize Redis for caching
    try:
        app.state.redis = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True,
            socket_connect_timeout=5
        )
        app.state.redis.ping()
        logger.info("Redis connected")
    except:
        logger.warning("Redis not available - caching disabled")
        app.state.redis = None
    
    # Initialize MongoDB for persistence
    try:
        app.state.mongodb = AsyncIOMotorClient('mongodb://localhost:27017')
        app.state.db = app.state.mongodb.dps_database
        logger.info("MongoDB connected")
    except:
        logger.warning("MongoDB not available - persistence disabled")
        app.state.mongodb = None
        app.state.db = None
    
    # Load default models
    await app.state.model_manager.initialize()
    
    logger.info("DPS API Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DPS API Server...")
    
    # Cleanup
    app.state.dps_core.shutdown()
    
    if app.state.mongodb:
        app.state.mongodb.close()
    
    if app.state.redis:
        app.state.redis.close()
    
    logger.info("DPS API Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Deep Parallel Synthesis API",
    description="Advanced scientific reasoning through parallel synthesis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Security
security = HTTPBearer()


# Dependency injection
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate JWT token and return current user"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, app.state.auth_manager.secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "Deep Parallel Synthesis API",
        "version": "2.0.0",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check(request: Request):
    """Health check endpoint"""
    uptime = time.time() - request.app.state.start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime=uptime,
        models_loaded=len(request.app.state.model_manager.loaded_models),
        active_tasks=request.app.state.dps_core.get_metrics().get("history_length", 0),
        redis_connected=request.app.state.redis is not None and request.app.state.redis.ping(),
        mongodb_connected=request.app.state.mongodb is not None
    )


@app.post("/auth/login", tags=["Authentication"])
async def login(username: str, password: str, request: Request):
    """Login endpoint"""
    user = await request.app.state.auth_manager.authenticate(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    token = request.app.state.auth_manager.create_token(user)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/reason", response_model=ReasoningResponse, tags=["Reasoning"])
async def reason(
    request: ReasoningRequest,
    background_tasks: BackgroundTasks,
    app_request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Main reasoning endpoint"""
    start_time = time.time()
    task_id = f"task_{int(time.time() * 1000)}"
    
    # Track metrics
    request_counter.labels(endpoint="/reason", method="POST").inc()
    reasoning_tasks.labels(status="started").inc()
    
    try:
        # Check cache if available
        if app_request.app.state.redis:
            cache_key = f"dps:{hash(request.prompt)}"
            cached = app_request.app.state.redis.get(cache_key)
            if cached:
                logger.info(f"Cache hit for prompt hash {hash(request.prompt)}")
                return json.loads(cached)
        
        # Convert reasoning types
        reasoning_types = None
        if request.reasoning_types:
            reasoning_types = [ReasoningType[rt.upper()] for rt in request.reasoning_types]
        
        # Configure DPS core
        dps = app_request.app.state.dps_core
        dps.num_chains = request.num_chains
        dps.max_depth = request.max_depth
        dps.synthesis_temperature = request.temperature
        
        # Select model backend
        model = await app_request.app.state.model_manager.get_model(request.model_backend)
        
        # Execute reasoning
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_reasoning(dps, request, task_id),
                media_type="text/event-stream"
            )
        else:
            # Regular response
            result = await dps.reason(
                prompt=request.prompt,
                context=request.context,
                reasoning_types=reasoning_types
            )
            
            # Validate if requested
            validation = None
            if request.validate:
                val_result = app_request.app.state.validator.validate(
                    content=result["response"],
                    reasoning_type="systematic",
                    evidence=result.get("evidence", [])
                )
                validation = val_result.to_dict()
            
            response = ReasoningResponse(
                id=task_id,
                response=result["response"],
                confidence=result["confidence"],
                reasoning_chains=result["reasoning_chains"],
                metrics=result["metrics"],
                evidence=result.get("evidence", []),
                validation=validation,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Cache result
            if app_request.app.state.redis:
                app_request.app.state.redis.setex(
                    cache_key,
                    300,  # 5 minute TTL
                    json.dumps(response.dict())
                )
            
            # Store in database
            if app_request.app.state.db:
                background_tasks.add_task(
                    store_reasoning_result,
                    app_request.app.state.db,
                    task_id,
                    request.dict(),
                    response.dict(),
                    current_user["username"]
                )
            
            # Track metrics
            duration = time.time() - start_time
            request_duration.labels(endpoint="/reason").observe(duration)
            reasoning_tasks.labels(status="completed").inc()
            
            return response
            
    except Exception as e:
        reasoning_tasks.labels(status="failed").inc()
        logger.error(f"Reasoning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reason/batch", tags=["Reasoning"])
async def batch_reason(
    batch: BatchRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Batch reasoning endpoint"""
    tasks = []
    for prompt in batch.prompts:
        req = batch.common_params.dict() if batch.common_params else {}
        req["prompt"] = prompt
        reasoning_request = ReasoningRequest(**req)
        
        task = asyncio.create_task(
            reason(reasoning_request, background_tasks, request, current_user)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        "total": len(batch.prompts),
        "successful": sum(1 for r in results if not isinstance(r, Exception)),
        "failed": sum(1 for r in results if isinstance(r, Exception)),
        "results": [
            r.dict() if not isinstance(r, Exception) else {"error": str(r)}
            for r in results
        ]
    }


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models(request: Request, current_user: dict = Depends(get_current_user)):
    """List available models"""
    models = await request.app.state.model_manager.list_models()
    return [
        ModelInfo(
            name=m["name"],
            backend=m["backend"],
            loaded=m["loaded"],
            memory_usage=m.get("memory_usage"),
            last_used=m.get("last_used")
        )
        for m in models
    ]


@app.post("/models/load", tags=["Models"])
async def load_model(
    model_name: str,
    backend: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Load a model"""
    start_time = time.time()
    
    try:
        await request.app.state.model_manager.load_model(model_name, backend)
        
        duration = time.time() - start_time
        model_load_time.observe(duration)
        
        return {
            "status": "loaded",
            "model": model_name,
            "backend": backend,
            "load_time": duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload", tags=["Models"])
async def unload_model(
    model_name: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Unload a model"""
    try:
        await request.app.state.model_manager.unload_model(model_name)
        return {"status": "unloaded", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}", tags=["Tasks"])
async def get_task(
    task_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get task result"""
    if not request.app.state.db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    result = await request.app.state.db.tasks.find_one({"task_id": task_id})
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result.pop("_id", None)
    return result


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get Prometheus metrics"""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/stats", tags=["Monitoring"])
async def get_stats(request: Request, current_user: dict = Depends(get_current_user)):
    """Get system statistics"""
    dps_metrics = request.app.state.dps_core.get_metrics()
    
    if request.app.state.db:
        total_tasks = await request.app.state.db.tasks.count_documents({})
        recent_tasks = await request.app.state.db.tasks.count_documents({
            "timestamp": {"$gte": (datetime.utcnow() - timedelta(hours=1)).isoformat()}
        })
    else:
        total_tasks = 0
        recent_tasks = 0
    
    return {
        "dps_metrics": dps_metrics,
        "total_tasks": total_tasks,
        "recent_tasks_1h": recent_tasks,
        "uptime": time.time() - request.app.state.start_time,
        "models_loaded": len(request.app.state.model_manager.loaded_models)
    }


@app.websocket("/ws/reason")
async def websocket_reasoning(websocket, current_user: dict = Depends(get_current_user)):
    """WebSocket endpoint for real-time reasoning"""
    await websocket.accept()
    active_connections.inc()
    
    try:
        while True:
            data = await websocket.receive_json()
            request = ReasoningRequest(**data)
            
            # Stream reasoning results
            async for chunk in stream_reasoning_ws(
                websocket.app.state.dps_core,
                request
            ):
                await websocket.send_json(chunk)
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.dec()
        await websocket.close()


# Helper functions

async def stream_reasoning(dps: DPSCore, request: ReasoningRequest, task_id: str):
    """Stream reasoning results"""
    # This would integrate with actual streaming from the model
    # For now, simulate streaming
    
    result = await dps.reason(
        prompt=request.prompt,
        context=request.context
    )
    
    # Stream chunks
    response_text = result["response"]
    chunk_size = 50
    
    for i in range(0, len(response_text), chunk_size):
        chunk = response_text[i:i+chunk_size]
        yield f"data: {json.dumps({'chunk': chunk, 'task_id': task_id})}\n\n"
        await asyncio.sleep(0.1)
    
    # Send final result
    yield f"data: {json.dumps({'done': True, 'confidence': result['confidence'], 'task_id': task_id})}\n\n"


async def stream_reasoning_ws(dps: DPSCore, request: ReasoningRequest):
    """Stream reasoning results over WebSocket"""
    result = await dps.reason(
        prompt=request.prompt,
        context=request.context
    )
    
    # Stream chunks
    response_text = result["response"]
    chunk_size = 50
    
    for i in range(0, len(response_text), chunk_size):
        chunk = response_text[i:i+chunk_size]
        yield {"type": "chunk", "content": chunk}
        await asyncio.sleep(0.05)
    
    # Send final result
    yield {
        "type": "complete",
        "confidence": result["confidence"],
        "chains": len(result["reasoning_chains"])
    }


async def store_reasoning_result(db, task_id: str, request: dict, response: dict, username: str):
    """Store reasoning result in database"""
    try:
        await db.tasks.insert_one({
            "task_id": task_id,
            "username": username,
            "request": request,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to store result: {e}")


def main():
    """Run the API server"""
    uvicorn.run(
        "dps_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()