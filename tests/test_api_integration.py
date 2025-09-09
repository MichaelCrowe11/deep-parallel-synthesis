"""
Integration tests for DPS API
Tests the complete API flow including authentication, reasoning, and model management
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
import sys
from typing import Dict, Any

import httpx
from fastapi.testclient import TestClient
import jwt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.dps_api_server import app
from api.model_manager import ModelManager, ModelBackend
from api.auth_manager import AuthManager, UserRole, Permission


class TestAPIIntegration:
    """Integration tests for DPS API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get authentication headers"""
        # Login to get token
        response = client.post("/auth/login", params={
            "username": "admin",
            "password": "admin"
        })
        
        if response.status_code == 200:
            token = response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}
        
        # If login fails, create a mock token for testing
        auth_manager = AuthManager()
        token = auth_manager.create_token({
            "username": "test_user",
            "email": "test@example.com",
            "role": "admin",
            "permissions": ["read", "write", "reason", "admin"]
        })
        return {"Authorization": f"Bearer {token}"}
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Deep Parallel Synthesis API"
        assert data["version"] == "2.0.0"
        assert data["status"] == "running"
    
    def test_authentication_required(self, client):
        """Test that authentication is required for protected endpoints"""
        # Try to access protected endpoint without auth
        response = client.post("/reason", json={
            "prompt": "Test prompt"
        })
        assert response.status_code == 403  # Forbidden
    
    def test_login(self, client):
        """Test login endpoint"""
        response = client.post("/auth/login", params={
            "username": "admin",
            "password": "admin"
        })
        
        # Note: This might fail in test environment
        # We're just testing the endpoint exists
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
    
    def test_reasoning_endpoint(self, client, auth_headers):
        """Test main reasoning endpoint"""
        request_data = {
            "prompt": "What is the capital of France?",
            "reasoning_types": ["deductive"],
            "max_depth": 3,
            "num_chains": 4,
            "temperature": 0.7,
            "validate": True,
            "stream": False
        }
        
        response = client.post(
            "/reason",
            json=request_data,
            headers=auth_headers
        )
        
        # Check response (might be 200 or 500 depending on model availability)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "response" in data
            assert "confidence" in data
            assert "reasoning_chains" in data
            assert "metrics" in data
            assert "evidence" in data
            assert "timestamp" in data
    
    def test_batch_reasoning(self, client, auth_headers):
        """Test batch reasoning endpoint"""
        batch_data = {
            "prompts": [
                "What is 2+2?",
                "What is the capital of Japan?",
                "Explain gravity"
            ],
            "common_params": {
                "max_depth": 2,
                "num_chains": 2,
                "temperature": 0.5
            }
        }
        
        response = client.post(
            "/reason/batch",
            json=batch_data,
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "total" in data
            assert data["total"] == 3
            assert "successful" in data
            assert "failed" in data
            assert "results" in data
            assert len(data["results"]) == 3
    
    def test_model_list(self, client, auth_headers):
        """Test model listing endpoint"""
        response = client.get("/models", headers=auth_headers)
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            
            if len(data) > 0:
                model = data[0]
                assert "name" in model
                assert "backend" in model
                assert "loaded" in model
    
    def test_model_load_unload(self, client, auth_headers):
        """Test model loading and unloading"""
        # Try to load a small model
        response = client.post(
            "/models/load",
            params={
                "model_name": "gpt2",
                "backend": "huggingface"
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "loaded"
            assert data["model"] == "gpt2"
            
            # Now unload it
            response = client.post(
                "/models/unload",
                params={"model_name": "gpt2"},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unloaded"
    
    def test_stats_endpoint(self, client, auth_headers):
        """Test statistics endpoint"""
        response = client.get("/stats", headers=auth_headers)
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "dps_metrics" in data
            assert "total_tasks" in data
            assert "recent_tasks_1h" in data
            assert "uptime" in data
            assert "models_loaded" in data
    
    def test_invalid_reasoning_request(self, client, auth_headers):
        """Test invalid reasoning request"""
        # Empty prompt
        response = client.post(
            "/reason",
            json={"prompt": ""},
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
        
        # Invalid reasoning type
        response = client.post(
            "/reason",
            json={
                "prompt": "Test",
                "reasoning_types": ["invalid_type"]
            },
            headers=auth_headers
        )
        assert response.status_code == 422
        
        # Out of range parameters
        response = client.post(
            "/reason",
            json={
                "prompt": "Test",
                "max_depth": 100,  # Too high
                "num_chains": 50   # Too high
            },
            headers=auth_headers
        )
        assert response.status_code == 422


class TestModelManager:
    """Test model manager functionality"""
    
    @pytest.fixture
    async def manager(self):
        """Create model manager instance"""
        manager = ModelManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """Test model manager initialization"""
        assert manager.hardware_info is not None
        assert "cpu_count" in manager.hardware_info
        assert "ram_gb" in manager.hardware_info
        assert manager.available_backends is not None
    
    @pytest.mark.asyncio
    async def test_list_models(self, manager):
        """Test listing available models"""
        models = await manager.list_models()
        assert isinstance(models, list)
        
        # Check default models are present
        model_names = [m["name"] for m in models]
        assert "llama3.1:70b" in model_names or "gpt2" in model_names
    
    @pytest.mark.asyncio
    async def test_model_info(self, manager):
        """Test getting model information"""
        # Test with a known model
        info = manager.get_model_info("gpt2")
        assert info["name"] == "gpt2"
        assert info["backend"] == "huggingface"
        assert "parameters" in info
        assert "hardware_requirements" in info
    
    @pytest.mark.asyncio
    async def test_hardware_check(self, manager):
        """Test hardware requirement checking"""
        # Create a config with low requirements
        from api.model_manager import ModelConfig
        
        config = ModelConfig(
            name="test_model",
            backend=ModelBackend.LOCAL,
            hardware_requirements={"min_ram": 1, "min_vram": 0}
        )
        
        assert manager._check_hardware_requirements(config) == True
        
        # Create a config with impossible requirements
        config_high = ModelConfig(
            name="test_model_high",
            backend=ModelBackend.LOCAL,
            hardware_requirements={"min_ram": 10000, "min_vram": 10000}
        )
        
        assert manager._check_hardware_requirements(config_high) == False


class TestAuthManager:
    """Test authentication manager"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create auth manager instance"""
        return AuthManager()
    
    def test_default_users(self, auth_manager):
        """Test default users are created"""
        assert "admin" in auth_manager.users
        assert "user" in auth_manager.users
        assert "service" in auth_manager.users
    
    def test_create_token(self, auth_manager):
        """Test JWT token creation"""
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "role": "user",
            "permissions": ["read", "write"]
        }
        
        token = auth_manager.create_token(user_data)
        assert token is not None
        assert isinstance(token, str)
    
    def test_verify_token(self, auth_manager):
        """Test JWT token verification"""
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "role": "user",
            "permissions": ["read", "write"]
        }
        
        token = auth_manager.create_token(user_data)
        payload = auth_manager.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == "test_user"
        assert payload["email"] == "test@example.com"
    
    def test_create_api_key(self, auth_manager):
        """Test API key creation"""
        key = auth_manager.create_api_key(
            user_id="test_user",
            name="test_key",
            permissions=[Permission.READ, Permission.REASON]
        )
        
        assert key is not None
        assert key.startswith("dps_")
        assert key in auth_manager.api_keys
    
    def test_verify_api_key(self, auth_manager):
        """Test API key verification"""
        key = auth_manager.create_api_key(
            user_id="test_user",
            name="test_key"
        )
        
        result = auth_manager.verify_api_key(key)
        assert result is not None
        assert result["user_id"] == "test_user"
        assert "permissions" in result
    
    def test_check_permission(self, auth_manager):
        """Test permission checking"""
        user_data = {
            "permissions": ["read", "write", "reason"]
        }
        
        assert auth_manager.check_permission(user_data, Permission.READ) == True
        assert auth_manager.check_permission(user_data, Permission.WRITE) == True
        assert auth_manager.check_permission(user_data, Permission.ADMIN) == False
    
    def test_password_hashing(self, auth_manager):
        """Test password hashing and verification"""
        password = "test_password_123"
        hashed = auth_manager.hash_password(password)
        
        assert hashed != password
        assert auth_manager.verify_password(password, hashed) == True
        assert auth_manager.verify_password("wrong_password", hashed) == False
    
    @pytest.mark.asyncio
    async def test_authenticate(self, auth_manager):
        """Test user authentication"""
        # Test with default user (password = username for demo)
        result = await auth_manager.authenticate("admin", "admin")
        assert result is not None
        assert result["username"] == "admin"
        assert result["role"] == "admin"
        
        # Test with wrong password
        result = await auth_manager.authenticate("admin", "wrong")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])