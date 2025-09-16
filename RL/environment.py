import torch 
from torch.nn import nn 
from dataclasses import dataclass
from .A2C import Actor , Critic
from typing import List , Any , Dict , Tuple , Optional
from collections import deque
import logging 
import numpy as np 
from Graph.data.community import Community , CommunityFeatures
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class State:
    """State containing LLM features and current selection status"""
    llm_query_features: torch.Tensor     
    llm_community_features: torch.Tensor 
    llm_hypergraph_features: torch.Tensor 
    available_communities: List[str]      
    selected_communities: List[str]      
    remaining_budget: int                 
    community_embeddings: torch.Tensor  
    community_scores: torch.Tensor       
    
    def to_flat_tensor(self) -> torch.Tensor:
        """Convert entire state to flat tensor for neural network input"""
        components = [
            self.llm_query_features.flatten(),
            self.llm_community_features.flatten(), 
            self.llm_hypergraph_features.flatten(),
            torch.tensor([self.remaining_budget / 10.0]),
            self.community_embeddings.flatten(), 
            self.community_scores  
        ]

        
        #selection mask (0 if selected, 1 if available)
        selection_mask = torch.tensor([
            0.0 if comm_id in self.selected_communities else 1.0 
            for comm_id in self.available_communities
        ])
        components.append(selection_mask)
        
        return torch.cat(components, dim=0)
    
@dataclass
class Action: 
    def create_action_space(self, pipeline_output: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Create action space tensors for RL agent
        
        Args:
            pipeline_output: Output from execute_tensorized_pipeline
            
        Returns:
            Dictionary of possible actions as tensors
        """
        actions = {}
        
        if 'communities' in pipeline_output:
            community_text = pipeline_output['communities']['text']
            community_mentions = community_text.lower().count('community')
            actions['select_communities'] = torch.arange(community_mentions, dtype=torch.float32)
        
        
        if 'hypergraph' in pipeline_output:
            hypergraph_text = pipeline_output['hypergraph']['text']
            edge_mentions = hypergraph_text.lower().count('edge')
            actions['navigate_hypergraph'] = torch.arange(max(1, edge_mentions), dtype=torch.float32)
        
        return actions
    

@dataclass
class Reward:

    def __init__()

        
    



        