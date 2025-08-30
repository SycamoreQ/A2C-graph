from storage import *
import sqlite3
import re 
import os 
import logging
from torch_geometric import * 
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple
import tiktoken

class QuestionResult: 
    response: str 
    context_data: str | dict[str, Any]
    completion_time : int 
    llm_calls : int 
    prompt_token : int 
    


class AuthorQuery: 
    
    def __init__(self ,
                 model: ChatModel,
                 token_encoder: tiktoken.Encoding,
                 model_params : dict[str, Any] = None | None,
                 context_size : int = 2048 , 
                 context_params : dict[str, Any] = None | None,
                 entity_type: str = "author"
                 ):
        self.model = model
        self.token_encoder = token_encoder
        self.model_params = model_params if model_params is not None else {}
        self.context_size = context_size
        self.context_params = context_params if context_params is not None else {}


    async def question_generate(
            self , 
            question_memory: list[str],
            context_data: str | dict[str , Any],
            question_count : str ,
            **kwargs 
    ) -> QuestionResult:
        "Generate a response "
    


class PubicationQuery: 
    
    def __init__(self,
                 model : ChatModel,
                 token_encoder: tiktoken.Encoding,
                 model_params: dict[str , Any] = None | None,
                 context_size: int = 2048,
                 context_params: dict[str , Any] = None | None ,
                 entity_type : str = "publication"
                   ):
        self.model = model 
        self.token_encoder = token_encoder
        self.model_params= model_params if model_params is not None else {}
        self.context_size = context_size
        self.context_params = context_params if context_params is not None else {}
        