import json 
from typing import Any 
import redis 
from caching.pipeline import PipelineCache
from redis.exceptions import RedisError
from redis_cache import RedisCache  

class JSONCache(PipelineCache):
    """A simple JSON file based cache implementation in redis."""
    
    def __init__(self , redis_cache: RedisCache):
        self.redis_cache = redis_cache
        self