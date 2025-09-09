"""Converts LLM response into input for A2C model"""

import json 
import torch 
from Graph.data.community import CommunityFeatures 
from typing import Any , Dict , List , Tuple , Optional
from Graph.data.entity import Entity
from Graph.data.relationship import Relationship
from dataclasses import dataclass
from RL.environment import State , Action , Environment , Reward 
from model.llm import PaperRetrievalSystem 
import tiktoken
from transformers import AutoTokenizer
from Graph.data.community import FEATURE_SCHEMA

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="original")


@dataclass
class TokenizeOutput: 
    text: str
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    embeddings: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None


@dataclass
class TensorizedFeatures:
    """Container for tensorized features extracted from text"""
    text_embeddings: torch.Tensor
    feature_vector: torch.Tensor
    categorical_features: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]


@dataclass
class ModelResponse:
    full_response: PaperRetrievalSystem 
    user_query: str

    def to_tensor(self , response : Dict[str , Any]) -> Dict[str , torch.Tensor] : 
        response = PaperRetrievalSystem.execute_community_selection_pipeline(user_query= self.user_query)
        response_values = response.values
        response_keys = response.keys

        for i in response_keys:
            if i == 'detected_communities' or 'hypergraph': 
                tokenizer.

    



     