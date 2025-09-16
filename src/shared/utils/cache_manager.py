"""
Cache manager for the Dynamic Pricing Intelligence System.
Provides Redis-based caching with TTL, invalidation, and performance monitoring.
"""

import json
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio

import redis.asyncio as redis
from redis.exceptions import RedisError

from .logging_utils import get_logger
from .error_handling import PricingIntelligenceError, ErrorCategory
from ..config.settings import settings


logger = get_logger(__name__)


class CacheError(PricingIntelligenceError):
    """Cache operation error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            **kwargs
        )


class CacheManager:
    """
    Redis-based cache manager with advanced features.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL (uses settings if not provided)
        """
        self.redis_url = redis_url or settings.cache.redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = settings.cache.default_ttl_seconds
        self.key_prefix = settings.cache.key_prefix
        
        # Performance metrics
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Cache manager connected to Redis")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise CacheError(f"Redis connection failed: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            logger.info("Cache manager disconnected from Redis")
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}:{key}"
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage."""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        else:
            # Use pickle for complex objects, then base64 encode
            import base64
            pickled = pickle.dumps(value)
            encoded = base64.b64encode(pickled).decode('utf-8')
            return f"pickle:{encoded}"
    
    def _deserialize_value(self, value: str) -> Any:
        """Deserialize value from storage."""
        if value.startswith("pickle:"):
            # Decode pickled object
            import base64
            encoded = value[7:]  # Remove "pickle:" prefix
            pickled = base64.b64decode(encoded.encode('utf-8'))
            return pickle.loads(pickled)
        else:
            return json.loads(value)
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return None
        
        try:
            cache_key = self._make_key(key)
            value = await self.redis_client.get(cache_key)
            
            if value is None:
                self.miss_count += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
            
            self.hit_count += 1
            logger.debug(f"Cache hit for key: {key}")
            
            return self._deserialize_value(value)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if not provided)
            nx: Only set if key doesn't exist
            
        Returns:
            True if value was set, False otherwise
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return False
        
        try:
            cache_key = self._make_key(key)
            serialized_value = self._serialize_value(value)
            ttl_seconds = ttl or self.default_ttl
            
            result = await self.redis_client.set(
                cache_key,
                serialized_value,
                ex=ttl_seconds,
                nx=nx
            )
            
            if result:
                logger.debug(f"Cache set for key: {key}, TTL: {ttl_seconds}s")
            
            return bool(result)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if key didn't exist
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.delete(cache_key)
            
            if result:
                logger.debug(f"Cache delete for key: {key}")
            
            return bool(result)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache exists error for key {key}: {str(e)}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if expiration was set, False if key doesn't exist
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.expire(cache_key, ttl)
            return bool(result)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache expire error for key {key}: {str(e)}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment numeric value in cache.
        
        Args:
            key: Cache key
            amount: Amount to increment by
            
        Returns:
            New value after increment, or None on error
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return None
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.incrby(cache_key, amount)
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache increment error for key {key}: {str(e)}")
            return None
    
    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs for found keys
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return {}
        
        try:
            cache_keys = [self._make_key(key) for key in keys]
            values = await self.redis_client.mget(cache_keys)
            
            result = {}
            for i, (key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    result[key] = self._deserialize_value(value)
                    self.hit_count += 1
                else:
                    self.miss_count += 1
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache get_multiple error: {str(e)}")
            return {}
    
    async def set_multiple(
        self,
        key_value_pairs: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            key_value_pairs: Dictionary of key-value pairs to set
            ttl: Time to live in seconds (uses default if not provided)
            
        Returns:
            True if all values were set successfully
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return False
        
        try:
            # Prepare data for mset
            cache_data = {}
            for key, value in key_value_pairs.items():
                cache_key = self._make_key(key)
                serialized_value = self._serialize_value(value)
                cache_data[cache_key] = serialized_value
            
            # Set all values
            await self.redis_client.mset(cache_data)
            
            # Set TTL for each key if specified
            if ttl:
                ttl_seconds = ttl or self.default_ttl
                for cache_key in cache_data.keys():
                    await self.redis_client.expire(cache_key, ttl_seconds)
            
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache set_multiple error: {str(e)}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            await self.connect()
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return 0
        
        try:
            cache_pattern = self._make_key(pattern)
            keys = await self.redis_client.keys(cache_pattern)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Cache clear_pattern error for pattern {pattern}: {str(e)}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
        
        stats = {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "error_count": self.error_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "connected": self.redis_client is not None
        }
        
        # Add Redis info if connected
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info()
                stats["redis_info"] = {
                    "used_memory": redis_info.get("used_memory"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "total_commands_processed": redis_info.get("total_commands_processed")
                }
            except Exception:
                pass
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache connection.
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            "status": "unhealthy",
            "connected": False,
            "latency_ms": None,
            "error": None
        }
        
        try:
            if not self.redis_client:
                await self.connect()
            
            # Measure ping latency
            start_time = datetime.now()
            if self.redis_client:
                await self.redis_client.ping()
            end_time = datetime.now()
            
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            health_status.update({
                "status": "healthy",
                "connected": True,
                "latency_ms": round(latency_ms, 2)
            })
            
        except Exception as e:
            health_status["error"] = str(e)
            logger.error(f"Cache health check failed: {str(e)}")
        
        return health_status
    
    def cache_result(
        self,
        key_func: Optional[Callable] = None,
        ttl: Optional[int] = None,
        skip_cache: bool = False
    ):
        """
        Decorator to cache function results.
        
        Args:
            key_func: Function to generate cache key (uses function name and args if not provided)
            ttl: Time to live in seconds
            skip_cache: Skip cache and always execute function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if skip_cache:
                    return await func(*args, **kwargs)
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend([str(arg) for arg in args])
                    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                    key_string = ":".join(key_parts)
                    cache_key = hashlib.md5(key_string.encode()).hexdigest()
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager


async def init_cache() -> CacheManager:
    """Initialize cache manager and establish connection."""
    cache_manager = get_cache_manager()
    await cache_manager.connect()
    return cache_manager


async def close_cache() -> None:
    """Close cache manager connection."""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.disconnect()
        _cache_manager = None
