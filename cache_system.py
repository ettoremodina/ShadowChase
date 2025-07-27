#!/usr/bin/env python3
"""
Persistent Cache System for Scotland Yard RL

A high-performance, persistent caching system that can be shared across:
- Game methods (get_valid_moves, is_game_over, etc.)
- Agent decision-making (MCTS nodes, evaluations, etc.)
- Training data and model states

Features:
- JSON-based persistence with automatic compression
- LRU eviction with configurable size limits
- TTL (time-to-live) support for cache expiration
- Multiple cache namespaces for different use cases
- Thread-safe operations
- Automatic cleanup and maintenance
- Performance monitoring and statistics
"""

import json
import gzip
import hashlib
import time
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import atexit


class CacheNamespace(Enum):
    """Cache namespaces for different types of data."""
    GAME_METHODS = "game_methods"
    MCTS_NODES = "mcts_nodes"
    AGENT_DECISIONS = "agent_decisions"
    TRAINING_DATA = "training_data"
    EVALUATIONS = "evaluations"


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    namespace: str
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class PersistentCache:
    """
    High-performance persistent cache system with multiple namespaces.
    
    This cache system is designed to handle:
    1. Game method results (get_valid_moves, is_game_over)
    2. MCTS node evaluations and tree search results
    3. Agent decision caching across games
    4. Training data and model evaluations
    """
    
    def __init__(self, 
                 cache_dir: str = "cache_data",
                 max_size_per_namespace: int = 5e7,
                 max_total_size_mb: float = 100.0,
                 auto_save_interval: float = 30.0,
                 compression_enabled: bool = True):
        """
        Initialize the persistent cache system.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_per_namespace: Maximum entries per namespace
            max_total_size_mb: Maximum total cache size in MB
            auto_save_interval: Auto-save interval in seconds
            compression_enabled: Whether to use gzip compression
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_size_per_namespace = max_size_per_namespace
        self.max_total_size_mb = max_total_size_mb
        self.auto_save_interval = auto_save_interval
        self.compression_enabled = compression_enabled
        
        # In-memory cache storage
        self._cache: Dict[str, Dict[str, CacheEntry]] = {
            namespace.value: {} for namespace in CacheNamespace
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'evictions': 0,
            'disk_saves': 0,
            'disk_loads': 0,
            'total_requests': 0
        }
        
        # Background maintenance
        self._last_save = time.time()
        self._shutdown = False
        
        # Load existing cache from disk
        self._load_from_disk()
        
        # Register cleanup on exit
        atexit.register(self._save_to_disk)
    
    def get(self, key: str, namespace: CacheNamespace, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        # Check if caching is globally disabled or namespace is disabled
        if not is_namespace_cache_enabled(namespace):
            self.stats['total_requests'] += 1
            self.stats['misses'] += 1
            return default
            
        with self._lock:
            self.stats['total_requests'] += 1
            
            namespace_cache = self._cache[namespace.value]
            
            if key in namespace_cache:
                entry = namespace_cache[key]
                
                # Update access info
                entry.touch()
                self.stats['hits'] += 1
                return entry.value
            
            self.stats['misses'] += 1
            return default
    
    def put(self, key: str, value: Any, namespace: CacheNamespace, ttl_seconds: Optional[float] = None):
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            namespace: Cache namespace
            ttl_seconds: Ignored (TTL not supported anymore)
        """
        # Check if caching is globally disabled or namespace is disabled
        if not is_namespace_cache_enabled(namespace):
            return
            
        with self._lock:
            namespace_cache = self._cache[namespace.value]
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                namespace=namespace.value
            )
            
            # Store in cache
            namespace_cache[key] = entry
            self.stats['writes'] += 1
            
            # Enforce size limits
            self._enforce_size_limits(namespace)
            
            # Auto-save if needed
            self._maybe_auto_save()
    
    def delete(self, key: str, namespace: CacheNamespace) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            namespace: Cache namespace
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            namespace_cache = self._cache[namespace.value]
            if key in namespace_cache:
                del namespace_cache[key]
                return True
            return False
    
    def clear_namespace(self, namespace: CacheNamespace):
        """Clear all entries in a namespace."""
        with self._lock:
            self._cache[namespace.value].clear()
    
    def clear_all(self):
        """Clear all cached data."""
        with self._lock:
            for namespace_cache in self._cache.values():
                namespace_cache.clear()
    
    def get_namespace_stats(self, namespace: CacheNamespace) -> Dict[str, Any]:
        """Get statistics for a specific namespace."""
        with self._lock:
            namespace_cache = self._cache[namespace.value]
            
            total_entries = len(namespace_cache)
            
            if total_entries > 0:
                avg_access_count = sum(entry.access_count for entry in namespace_cache.values()) / total_entries
                oldest_entry = min(namespace_cache.values(), key=lambda e: e.created_at)
                newest_entry = max(namespace_cache.values(), key=lambda e: e.created_at)
                age_range = newest_entry.created_at - oldest_entry.created_at
            else:
                avg_access_count = 0
                age_range = 0
            
            return {
                'total_entries': total_entries,
                'valid_entries': total_entries,  # All entries are valid (no expiration)
                'avg_access_count': avg_access_count,
                'age_range_seconds': age_range,
                'max_size': self.max_size_per_namespace
            }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cache statistics."""
        with self._lock:
            total_requests = self.stats['total_requests']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            total_entries = sum(len(ns_cache) for ns_cache in self._cache.values())
            cache_size_mb = self._estimate_cache_size_mb()
            
            namespace_stats = {}
            for namespace in CacheNamespace:
                namespace_stats[namespace.value] = self.get_namespace_stats(namespace)
            
            return {
                'global_stats': {
                    'hit_rate': hit_rate,
                    'total_entries': total_entries,
                    'cache_size_mb': cache_size_mb,
                    'max_size_mb': self.max_total_size_mb,
                    **self.stats
                },
                'namespace_stats': namespace_stats
            }
    
    def create_game_cache_key(self, method_name: str, **kwargs) -> str:
        """
        Create a cache key for game methods.
        
        Args:
            method_name: Name of the game method (e.g., 'get_valid_moves')
            **kwargs: Method parameters
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        
        # Create a string representation
        params_str = json.dumps(sorted_params, sort_keys=True, default=str)
        
        # Create hash
        key_data = f"{method_name}:{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def create_agent_cache_key(self, agent_type: str, agent_id: str, **kwargs) -> str:
        """
        Create a cache key for agent decisions.
        
        Args:
            agent_type: Type of agent (e.g., 'mcts', 'neural', etc.)
            agent_id: Unique agent identifier
            **kwargs: Decision parameters
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        
        # Create a string representation
        params_str = json.dumps(sorted_params, sort_keys=True, default=str)
        
        # Create hash
        key_data = f"{agent_type}:{agent_id}:{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _enforce_size_limits(self, namespace: CacheNamespace):
        """Enforce size limits using LRU eviction."""
        namespace_cache = self._cache[namespace.value]
        
        # Check namespace size limit
        while len(namespace_cache) > self.max_size_per_namespace:
            # Find LRU entry
            lru_key = min(namespace_cache.keys(), 
                         key=lambda k: namespace_cache[k].last_accessed)
            del namespace_cache[lru_key]
            self.stats['evictions'] += 1
        
        # Check total cache size
        if self._estimate_cache_size_mb() > self.max_total_size_mb:
            self._global_lru_eviction()
    
    def _global_lru_eviction(self):
        """Perform global LRU eviction across all namespaces."""
        # Collect all entries with their last access times
        all_entries = []
        for namespace_name, namespace_cache in self._cache.items():
            for key, entry in namespace_cache.items():
                all_entries.append((namespace_name, key, entry.last_accessed))
        
        # Sort by last accessed time
        all_entries.sort(key=lambda x: x[2])
        
        # Remove oldest 10% of entries
        entries_to_remove = max(1, len(all_entries) // 10)
        
        for i in range(entries_to_remove):
            namespace_name, key, _ = all_entries[i]
            if key in self._cache[namespace_name]:
                del self._cache[namespace_name][key]
                self.stats['evictions'] += 1
    
    def _estimate_cache_size_mb(self) -> float:
        """Estimate cache size in MB (rough approximation)."""
        # This is a rough estimate based on number of entries
        # In practice, you might want to implement more accurate size tracking
        total_entries = sum(len(ns_cache) for ns_cache in self._cache.values())
        # Assume average 1KB per entry (rough estimate)
        return total_entries * 1024 / (1024 * 1024)
    
    def _maybe_auto_save(self):
        """Auto-save cache to disk if interval has passed."""
        if time.time() - self._last_save > self.auto_save_interval:
            self._save_to_disk()
    
    def _save_to_disk(self):
        """Save cache to disk."""
        if self._shutdown:
            return
        
        try:
            for namespace in CacheNamespace:
                namespace_cache = self._cache[namespace.value]
                if not namespace_cache:
                    continue
                
                # Prepare data for serialization
                cache_data = {
                    'entries': {},
                    'metadata': {
                        'saved_at': time.time(),
                        'namespace': namespace.value,
                        'total_entries': len(namespace_cache)
                    }
                }
                
                # Convert entries to serializable format
                for key, entry in namespace_cache.items():
                    cache_data['entries'][key] = asdict(entry)
                
                # Save to file
                filename = f"cache_{namespace.value}.json"
                if self.compression_enabled:
                    filename += ".gz"
                
                filepath = self.cache_dir / filename
                
                if self.compression_enabled:
                    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, default=str)
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, default=str)
            
            self.stats['disk_saves'] += 1
            self._last_save = time.time()
            
        except Exception as e:
            print(f"Warning: Failed to save cache to disk: {e}")
    
    def _load_from_disk(self):
        """Load cache from disk."""
        try:
            for namespace in CacheNamespace:
                filename = f"cache_{namespace.value}.json"
                if self.compression_enabled:
                    filename += ".gz"
                
                filepath = self.cache_dir / filename
                
                if not filepath.exists():
                    continue
                
                # Load data
                if self.compression_enabled:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        cache_data = json.load(f)
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                
                # Restore entries
                namespace_cache = self._cache[namespace.value]
                entries_loaded = 0
                
                for key, entry_data in cache_data.get('entries', {}).items():
                    # Recreate CacheEntry object
                    entry = CacheEntry(
                        key=entry_data['key'],
                        value=entry_data['value'],
                        created_at=float(entry_data['created_at']),
                        last_accessed=float(entry_data['last_accessed']),
                        access_count=int(entry_data['access_count']),
                        namespace=entry_data['namespace']
                    )
                    
                    # Load all entries (no expiration check)
                    namespace_cache[key] = entry
                    entries_loaded += 1
                
                self.stats['disk_loads'] += 1
                
                print(f"✅ Loaded {entries_loaded} cache entries for namespace {namespace.value}")
        
        except Exception as e:
            print(f"Warning: Failed to load cache from disk: {e}")
    
    def shutdown(self):
        """Shutdown cache system and save to disk."""
        self._shutdown = True
        self._save_to_disk()


# Global cache instance and settings
_global_cache: Optional[PersistentCache] = None
_cache_enabled: bool = True  # Global flag to enable/disable caching
_namespace_enabled: Dict[str, bool] = {}  # Per-namespace enable/disable flags


def get_global_cache() -> PersistentCache:
    """Get the global cache instance, creating it if necessary."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PersistentCache()
    return _global_cache


def init_cache(cache_dir: str = "cache_data", **kwargs) -> PersistentCache:
    """Initialize the global cache with custom settings."""
    global _global_cache
    _global_cache = PersistentCache(cache_dir=cache_dir, **kwargs)
    return _global_cache


def enable_cache():
    """Enable global caching."""
    global _cache_enabled
    _cache_enabled = True
    print("✅ Cache enabled globally")


def disable_cache():
    """Disable global caching (cache operations become no-ops)."""
    global _cache_enabled
    _cache_enabled = False
    print("❌ Cache disabled globally")


def is_cache_enabled() -> bool:
    """Check if caching is currently enabled."""
    return _cache_enabled


def enable_namespace_cache(namespace: CacheNamespace):
    """Enable caching for a specific namespace."""
    global _namespace_enabled
    _namespace_enabled[namespace.value] = True
    print(f"✅ Cache enabled for namespace: {namespace.value}")


def disable_namespace_cache(namespace: CacheNamespace):
    """Disable caching for a specific namespace."""
    global _namespace_enabled
    _namespace_enabled[namespace.value] = False
    print(f"❌ Cache disabled for namespace: {namespace.value}")


def is_namespace_cache_enabled(namespace: CacheNamespace) -> bool:
    """Check if caching is enabled for a specific namespace."""
    global _namespace_enabled
    # If global cache is disabled, namespace is disabled too
    if not _cache_enabled:
        return False
    # If namespace is explicitly disabled, return False
    if namespace.value in _namespace_enabled:
        return _namespace_enabled[namespace.value]
    # Default to enabled if not explicitly set
    return True


def reset_namespace_cache_settings():
    """Reset all namespace cache settings to default (enabled)."""
    global _namespace_enabled
    _namespace_enabled.clear()
    print("🔄 All namespace cache settings reset to default (enabled)")


def get_cache_status() -> Dict[str, Any]:
    """Get detailed cache status for all namespaces."""
    status = {
        'global_enabled': _cache_enabled,
        'namespaces': {}
    }
    
    for namespace in CacheNamespace:
        status['namespaces'][namespace.value] = {
            'enabled': is_namespace_cache_enabled(namespace),
            'explicitly_set': namespace.value in _namespace_enabled
        }
    
    return status


# Convenience decorators for easy caching
def cache_game_method(namespace: CacheNamespace = CacheNamespace.GAME_METHODS, 
                     ttl_seconds: Optional[float] = None):
    """
    Decorator to cache game method results.
    
    Args:
        namespace: Cache namespace to use
        ttl_seconds: Time-to-live for cached results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            
            # Create cache key
            method_name = func.__name__
            # Include 'self' object state in key for instance methods
            if args and hasattr(args[0], '__dict__'):
                # For instance methods, include relevant state
                self_obj = args[0]
                if hasattr(self_obj, 'game_state') and self_obj.game_state:
                    # For game objects, use game state for key
                    state_dict = {
                        'detective_positions': getattr(self_obj.game_state, 'detective_positions', []),
                        'mrx_position': getattr(self_obj.game_state, 'MrX_position', None),
                        'turn': getattr(self_obj.game_state, 'turn', None),
                        'turn_count': getattr(self_obj.game_state, 'turn_count', 0)
                    }
                    kwargs['_game_state'] = state_dict
            
            cache_key = cache.create_game_cache_key(method_name, args=args[1:], kwargs=kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key, namespace)
            if result is not None:
                return result
            
            # Cache miss - compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result, namespace, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


def cache_agent_decision(agent_type: str, agent_id: str, 
                        namespace: CacheNamespace = CacheNamespace.AGENT_DECISIONS,
                        ttl_seconds: Optional[float] = None):
    """
    Decorator to cache agent decision results.
    
    Args:
        agent_type: Type of agent (e.g., 'mcts')
        agent_id: Unique agent identifier
        namespace: Cache namespace to use
        ttl_seconds: Time-to-live for cached results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            
            # Create cache key
            cache_key = cache.create_agent_cache_key(agent_type, agent_id, args=args[1:], kwargs=kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key, namespace)
            if result is not None:
                return result
            
            # Cache miss - compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result, namespace, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator
