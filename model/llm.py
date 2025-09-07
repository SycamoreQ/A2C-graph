import json 
from utils import query_chat_openai
import redis
from redis.commands.search.field import TagField
from redis.commands.search.index_definition import IndexType , IndexDefinition
from redis.commands.search import Search 
from redis.commands.search.aggregation import AggregateRequest
from typing import List, Dict, Optional, Any



PAPER_RETRIEVAL_SYSTEM_PROMPT = """ You are an agent that """




