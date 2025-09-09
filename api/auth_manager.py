"""
Authentication and Authorization Manager for DPS API
Handles JWT tokens, user management, and API key authentication
"""

import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum

import jwt
import bcrypt
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    USER = "user"
    API_KEY = "api_key"
    SERVICE = "service"


class Permission(Enum):
    """API permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    REASON = "reason"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"


@dataclass
class User:
    """User model"""
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIKey:
    """API Key model"""
    key: str
    name: str
    user_id: str
    permissions: List[Permission]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rate_limit: int = 1000  # requests per hour
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthManager:
    """
    Authentication and authorization manager
    Handles user management, JWT tokens, and API keys
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.secret_key = self._get_or_create_secret_key()
        self.algorithm = "HS256"
        self.token_expiry_hours = 24
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.db: Optional[Any] = None
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [p for p in Permission],
            UserRole.USER: [
                Permission.READ,
                Permission.WRITE,
                Permission.REASON
            ],
            UserRole.API_KEY: [
                Permission.READ,
                Permission.REASON
            ],
            UserRole.SERVICE: [
                Permission.READ,
                Permission.WRITE,
                Permission.REASON,
                Permission.MODEL_LOAD,
                Permission.MODEL_UNLOAD
            ]
        }
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        else:
            self._create_default_users()
    
    def _get_or_create_secret_key(self) -> str:
        """Get or create JWT secret key"""
        secret_file = Path(".secret_key")
        if secret_file.exists():
            return secret_file.read_text().strip()
        else:
            secret = secrets.token_urlsafe(32)
            secret_file.write_text(secret)
            return secret
    
    def _load_config(self, config_path: str):
        """Load authentication configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load users
        for user_data in config.get("users", []):
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                role=UserRole(user_data["role"]),
                permissions=[Permission(p) for p in user_data.get("permissions", [])],
                is_active=user_data.get("is_active", True)
            )
            self.users[user.username] = user
        
        # Load API keys
        for key_data in config.get("api_keys", []):
            api_key = APIKey(
                key=key_data["key"],
                name=key_data["name"],
                user_id=key_data["user_id"],
                permissions=[Permission(p) for p in key_data.get("permissions", [])],
                rate_limit=key_data.get("rate_limit", 1000),
                is_active=key_data.get("is_active", True)
            )
            self.api_keys[api_key.key] = api_key
    
    def _create_default_users(self):
        """Create default users for development"""
        # Admin user
        self.users["admin"] = User(
            username="admin",
            email="admin@dps.local",
            role=UserRole.ADMIN,
            permissions=self.role_permissions[UserRole.ADMIN]
        )
        
        # Default user
        self.users["user"] = User(
            username="user",
            email="user@dps.local",
            role=UserRole.USER,
            permissions=self.role_permissions[UserRole.USER]
        )
        
        # Service account
        self.users["service"] = User(
            username="service",
            email="service@dps.local",
            role=UserRole.SERVICE,
            permissions=self.role_permissions[UserRole.SERVICE]
        )
        
        # Create default API key
        default_key = self.create_api_key(
            user_id="service",
            name="default_api_key",
            permissions=self.role_permissions[UserRole.API_KEY]
        )
        logger.info(f"Default API key created: {default_key}")
    
    async def initialize_db(self, mongodb_client: AsyncIOMotorClient):
        """Initialize database connection"""
        self.db = mongodb_client.dps_database
        
        # Create indexes
        await self.db.users.create_index("username", unique=True)
        await self.db.users.create_index("email", unique=True)
        await self.db.api_keys.create_index("key", unique=True)
        
        # Load users from database
        await self._load_users_from_db()
        await self._load_api_keys_from_db()
    
    async def _load_users_from_db(self):
        """Load users from database"""
        if not self.db:
            return
        
        async for user_doc in self.db.users.find():
            user = User(
                username=user_doc["username"],
                email=user_doc["email"],
                role=UserRole(user_doc["role"]),
                permissions=[Permission(p) for p in user_doc["permissions"]],
                created_at=user_doc.get("created_at", datetime.utcnow()),
                last_login=user_doc.get("last_login"),
                is_active=user_doc.get("is_active", True)
            )
            self.users[user.username] = user
    
    async def _load_api_keys_from_db(self):
        """Load API keys from database"""
        if not self.db:
            return
        
        async for key_doc in self.db.api_keys.find():
            api_key = APIKey(
                key=key_doc["key"],
                name=key_doc["name"],
                user_id=key_doc["user_id"],
                permissions=[Permission(p) for p in key_doc["permissions"]],
                created_at=key_doc.get("created_at", datetime.utcnow()),
                last_used=key_doc.get("last_used"),
                expires_at=key_doc.get("expires_at"),
                rate_limit=key_doc.get("rate_limit", 1000),
                is_active=key_doc.get("is_active", True)
            )
            self.api_keys[api_key.key] = api_key
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    async def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username and password"""
        # For demo, accept any password matching username
        # In production, use proper password hashing
        if username in self.users and password == username:
            user = self.users[username]
            
            if not user.is_active:
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            
            # Save to database if available
            if self.db:
                await self.db.users.update_one(
                    {"username": username},
                    {"$set": {"last_login": user.last_login}},
                    upsert=True
                )
            
            return {
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions]
            }
        
        return None
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token"""
        expiry = datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        
        payload = {
            "sub": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"],
            "permissions": user_data["permissions"],
            "exp": expiry,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: Optional[List[Permission]] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """Create new API key"""
        key = f"dps_{secrets.token_urlsafe(32)}"
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key=key,
            name=name,
            user_id=user_id,
            permissions=permissions or self.role_permissions[UserRole.API_KEY],
            expires_at=expires_at
        )
        
        self.api_keys[key] = api_key
        
        # Add to user's API keys
        if user_id in self.users:
            self.users[user_id].api_keys.append(key)
        
        # Save to database if available
        if self.db:
            asyncio.create_task(self._save_api_key_to_db(api_key))
        
        return key
    
    async def _save_api_key_to_db(self, api_key: APIKey):
        """Save API key to database"""
        if not self.db:
            return
        
        await self.db.api_keys.update_one(
            {"key": api_key.key},
            {"$set": {
                "key": api_key.key,
                "name": api_key.name,
                "user_id": api_key.user_id,
                "permissions": [p.value for p in api_key.permissions],
                "created_at": api_key.created_at,
                "expires_at": api_key.expires_at,
                "rate_limit": api_key.rate_limit,
                "is_active": api_key.is_active
            }},
            upsert=True
        )
    
    def verify_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Verify API key"""
        if key not in self.api_keys:
            return None
        
        api_key = self.api_keys[key]
        
        if not api_key.is_active:
            return None
        
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            return None
        
        # Update last used
        api_key.last_used = datetime.utcnow()
        
        # Save to database if available
        if self.db:
            asyncio.create_task(
                self.db.api_keys.update_one(
                    {"key": key},
                    {"$set": {"last_used": api_key.last_used}}
                )
            )
        
        return {
            "api_key": key,
            "user_id": api_key.user_id,
            "permissions": [p.value for p in api_key.permissions],
            "rate_limit": api_key.rate_limit
        }
    
    def check_permission(
        self,
        user_data: Dict[str, Any],
        required_permission: Permission
    ) -> bool:
        """Check if user has required permission"""
        user_permissions = user_data.get("permissions", [])
        return required_permission.value in user_permissions
    
    def check_rate_limit(self, api_key: str, redis_client) -> bool:
        """Check API key rate limit"""
        if api_key not in self.api_keys:
            return False
        
        key_obj = self.api_keys[api_key]
        rate_limit = key_obj.rate_limit
        
        if not redis_client:
            return True  # No rate limiting without Redis
        
        # Rate limiting key
        rate_key = f"rate_limit:{api_key}:{datetime.utcnow().hour}"
        
        # Get current count
        current = redis_client.get(rate_key)
        if current:
            current = int(current)
            if current >= rate_limit:
                return False
        
        # Increment counter
        redis_client.incr(rate_key)
        redis_client.expire(rate_key, 3600)  # Expire after 1 hour
        
        return True
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> User:
        """Create new user"""
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        user = User(
            username=username,
            email=email,
            role=role,
            permissions=self.role_permissions[role]
        )
        
        self.users[username] = user
        
        # Save to database if available
        if self.db:
            password_hash = self.hash_password(password)
            await self.db.users.insert_one({
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "role": role.value,
                "permissions": [p.value for p in user.permissions],
                "created_at": user.created_at,
                "is_active": user.is_active
            })
        
        return user
    
    async def delete_user(self, username: str):
        """Delete user"""
        if username not in self.users:
            raise ValueError(f"User {username} not found")
        
        # Delete user's API keys
        user = self.users[username]
        for key in user.api_keys:
            if key in self.api_keys:
                del self.api_keys[key]
        
        # Delete user
        del self.users[username]
        
        # Delete from database
        if self.db:
            await self.db.users.delete_one({"username": username})
            await self.db.api_keys.delete_many({"user_id": username})
    
    def revoke_api_key(self, key: str):
        """Revoke API key"""
        if key in self.api_keys:
            self.api_keys[key].is_active = False
            
            # Update in database
            if self.db:
                asyncio.create_task(
                    self.db.api_keys.update_one(
                        {"key": key},
                        {"$set": {"is_active": False}}
                    )
                )
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users"""
        return [
            {
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "is_active": user.is_active,
                "api_key_count": len(user.api_keys)
            }
            for user in self.users.values()
        ]
    
    def list_api_keys(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List API keys"""
        keys = self.api_keys.values()
        
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        
        return [
            {
                "name": key.name,
                "user_id": key.user_id,
                "created_at": key.created_at.isoformat(),
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "rate_limit": key.rate_limit,
                "is_active": key.is_active
            }
            for key in keys
        ]