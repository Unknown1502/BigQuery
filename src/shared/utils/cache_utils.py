"""
Cache Utilities - Advanced caching and performance optimization
Handles Redis caching, in-memory caching, and cache invalidation strategies
"""

import asyncio
import json
import logging
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
import redis.asyncio as redis
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.config.settings_fixed import settings
from src.shared.utils.logging_utils import setup_logging
from src.shared.utils.error_handling_fixed import handle_exceptions, PricingIntelligenceError

# Setup logging
logger = setup_logging(__name__)

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    value: T
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

class InMemoryCache:
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                await self._remove(key)
                return None
            
            # Update access metadata
            entry.touch()
            
            # Move to end of access order (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> bool:
        """Set value in cache"""
        async with self._lock:
            # Calculate expiration
            expires_at = None
            if ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                expires_at = datetime.utcnow() + timedelta(seconds=self.default_ttl)
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                expires_at=expires_at,
                tags=tags or []
            )
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            # Store entry
            self._cache[key] = entry
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            return await self._remove(key)
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            return True
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags"""
        async with self._lock:
            keys_to_remove = []
            
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                await self._remove(key)
            
            return len(keys_to_remove)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'max_size': self.max_size,
                'utilization': (total_entries / self.max_size) * 100 if self.max_size > 0 else 0,
                'access_order_length': len(self._access_order)
            }
    
    async def _remove(self, key: str) -> bool:
        """Remove key from cache (internal method)"""
        if key in self._cache:
            del self._cache[key]
        
        if key in self._access_order:
            self._access_order.remove(key)
        
        return True
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if self._access_order:
            lru_key = self._access_order[0]
            await self._remove(lru_key)

class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        self.redis_url = redis_url or "redis://localhost:6379"
        self.default_ttl = default_ttl
        self._redis: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self._connected = False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self._connected or not self._redis:
            return None
        
        try:
            value = await self._redis.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON, fall back to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self._connected or not self._redis:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            # Set with TTL
            ttl_seconds = ttl or self.default_ttl
            await self._redis.setex(key, ttl_seconds, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not self._connected or not self._redis:
            return False
        
        try:
            result = await self._redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not self._connected or not self._redis:
            return 0
        
        try:
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing Redis pattern: {e}")
            return 0
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment counter in Redis"""
        if not self._connected or not self._redis:
            return 0
        
        try:
            result = await self._redis.incrby(key, amount)
            
            # Set TTL if this is a new key
            if result == amount and ttl:
                await self._redis.expire(key, ttl)
            
            return result
        except Exception as e:
            logger.error(f"Error incrementing Redis counter: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self._connected or not self._redis:
            return {'connected': False}
        
        try:
            info = await self._redis.info()
            return {
                'connected': True,
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1))
                ) * 100
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {'connected': False, 'error': str(e)}

class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (Redis) tiers"""
    
    def __init__(self, l1_size: int = 1000, l1_ttl: int = 300, 
                 l2_ttl: int = 3600, redis_url: Optional[str] = None):
        self.l1_cache = InMemoryCache(max_size=l1_size, default_ttl=l1_ttl)
        self.l2_cache = RedisCache(redis_url=redis_url, default_ttl=l2_ttl)
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'l1_sets': 0,
            'l2_sets': 0
        }
    
    async def connect(self):
        """Connect to Redis L2 cache"""
        await self.l2_cache.connect()
    
    async def disconnect(self):
        """Disconnect from Redis L2 cache"""
        await self.l2_cache.disconnect()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.stats['l1_hits'] += 1
            return value
        
        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            self.stats['l2_hits'] += 1
            # Promote to L1 cache
            await self.l1_cache.set(key, value)
            return value
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, l1_ttl: Optional[int] = None, 
                 l2_ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> bool:
        """Set value in multi-level cache"""
        # Set in both caches
        l1_success = await self.l1_cache.set(key, value, ttl=l1_ttl, tags=tags)
        l2_success = await self.l2_cache.set(key, value, ttl=l2_ttl)
        
        if l1_success:
            self.stats['l1_sets'] += 1
        if l2_success:
            self.stats['l2_sets'] += 1
        
        return l1_success or l2_success
    
    async def delete(self, key: str) -> bool:
        """Delete from both cache levels"""
        l1_result = await self.l1_cache.delete(key)
        l2_result = await self.l2_cache.delete(key)
        return l1_result or l2_result
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate by tags (L1 only, as Redis doesn't support tags natively)"""
        return await self.l1_cache.invalidate_by_tags(tags)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        l1_stats = await self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats()
        
        total_requests = sum([self.stats['l1_hits'], self.stats['l2_hits'], self.stats['misses']])
        
        return {
            'requests': {
                'total': total_requests,
                'l1_hits': self.stats['l1_hits'],
                'l2_hits': self.stats['l2_hits'],
                'misses': self.stats['misses'],
                'l1_hit_rate': (self.stats['l1_hits'] / total_requests * 100) if total_requests > 0 else 0,
                'l2_hit_rate': (self.stats['l2_hits'] / total_requests * 100) if total_requests > 0 else 0,
                'overall_hit_rate': ((self.stats['l1_hits'] + self.stats['l2_hits']) / total_requests * 100) if total_requests > 0 else 0
            },
            'l1_cache': l1_stats,
            'l2_cache': l2_stats
        }

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments"""
    # Create a deterministic key from arguments
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # Hash complex objects
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
    
    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}:{v}")
        else:
            key_parts.append(f"{k}:{hashlib.md5(str(v).encode()).hexdigest()[:8]}")
    
    return ":".join(key_parts)

def cached(ttl: int = 3600, cache_instance: Optional[MultiLevelCache] = None, 
          tags: Optional[List[str]] = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Use global cache if none provided
            cache = cache_instance or global_cache
            
            # Generate cache key
            func_key = f"{key_prefix}{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await cache.get(func_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(func_key, result, l1_ttl=min(ttl, 300), l2_ttl=ttl, tags=tags)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we need to handle caching differently
            # This is a simplified version - in production, you might want async cache operations
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Specialized caches for different data types
class PricingCache:
    """Specialized cache for pricing data"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.prefix = "pricing:"
    
    async def get_price(self, location_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get cached pricing data"""
        key = f"{self.prefix}price:{location_id}:{timestamp.strftime('%Y%m%d%H')}"
        return await self.cache.get(key)
    
    async def set_price(self, location_id: str, timestamp: datetime, 
                       pricing_data: Dict[str, Any], ttl: int = 1800) -> bool:
        """Cache pricing data"""
        key = f"{self.prefix}price:{location_id}:{timestamp.strftime('%Y%m%d%H')}"
        tags = [f"location:{location_id}", "pricing"]
        return await self.cache.set(key, pricing_data, l1_ttl=300, l2_ttl=ttl, tags=tags)
    
    async def invalidate_location(self, location_id: str) -> int:
        """Invalidate all pricing data for a location"""
        return await self.cache.invalidate_by_tags([f"location:{location_id}"])

class ImageCache:
    """Specialized cache for image analysis results"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.prefix = "image:"
    
    async def get_analysis(self, image_uri: str) -> Optional[Dict[str, Any]]:
        """Get cached image analysis"""
        key = f"{self.prefix}analysis:{hashlib.md5(image_uri.encode()).hexdigest()}"
        return await self.cache.get(key)
    
    async def set_analysis(self, image_uri: str, analysis_data: Dict[str, Any], 
                          ttl: int = 7200) -> bool:
        """Cache image analysis results"""
        key = f"{self.prefix}analysis:{hashlib.md5(image_uri.encode()).hexdigest()}"
        tags = ["image_analysis"]
        return await self.cache.set(key, analysis_data, l1_ttl=600, l2_ttl=ttl, tags=tags)

class CacheManager:
    """
    Unified cache manager that provides a simple interface to the multi-level cache system.
    This class is used by other services to interact with the cache.
    """
    
    def __init__(self, cache: Optional[MultiLevelCache] = None):
        self.cache = cache or global_cache
        self._connected = False
    
    async def connect(self):
        """Connect to cache systems"""
        await self.cache.connect()
        self._connected = True
        logger.info("CacheManager connected")
    
    async def disconnect(self):
        """Disconnect from cache systems"""
        await self.cache.disconnect()
        self._connected = False
        logger.info("CacheManager disconnected")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._connected:
            return None
        return await self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> bool:
        """Set value in cache"""
        if not self._connected:
            return False
        return await self.cache.set(key, value, l1_ttl=min(ttl or 300, 300), l2_ttl=ttl or 3600, tags=tags)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._connected:
            return False
        return await self.cache.delete(key)
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not self._connected:
            return 0
        # For L2 cache (Redis)
        return await self.cache.l2_cache.clear_pattern(pattern)
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags"""
        if not self._connected:
            return 0
        return await self.cache.invalidate_by_tags(tags)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache systems"""
        if not self._connected:
            return {"status": "disconnected", "healthy": False}
        
        try:
            stats = await self.cache.get_stats()
            return {
                "status": "connected",
                "healthy": True,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": "error",
                "healthy": False,
                "error": str(e)
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._connected:
            return {"connected": False}
        return await self.cache.get_stats()

# Global cache instance
global_cache = MultiLevelCache(
    l1_size=1000,
    l1_ttl=300,  # 5 minutes
    l2_ttl=3600,  # 1 hour
    redis_url=getattr(settings, 'redis_url', None)
)

# Global cache manager instance
cache_manager = CacheManager(global_cache)

# Specialized cache instances
pricing_cache = PricingCache(global_cache)
image_cache = ImageCache(global_cache)

async def initialize_cache():
    """Initialize cache connections"""
    await global_cache.connect()
    await cache_manager.connect()
    logger.info("Cache system initialized")
=======
from src.shared.utils.cache_manager import CacheManager
