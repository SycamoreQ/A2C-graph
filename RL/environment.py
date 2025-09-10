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
class Action:
    """Represents an action of selecting a community."""
    community_id: str
    selection_probability: float
    action_index: int


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
    def create_rl_action_space(self, pipeline_output: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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
        
        if 'refined_query' in pipeline_output:
            refinement_text = pipeline_output['refined_query']['text']
            actions['refine_query'] = torch.tensor([1.0, 0.5, 0.0])  # [accept, modify, reject]
        
        if 'hypergraph' in pipeline_output:
            hypergraph_text = pipeline_output['hypergraph']['text']
            edge_mentions = hypergraph_text.lower().count('edge')
            actions['navigate_hypergraph'] = torch.arange(max(1, edge_mentions), dtype=torch.float32)
        
        return actions

        

@dataclass 
class Environment: 

    def __init__(self , 
                 max_selections = 5 ,
                 relevance_weight = 0.5 , 
                 quality_weight = 0.7 , 
                 diversity_weight = 0.3):
        
        self.max_selections = max_selections
        self.relevance_weight = relevance_weight
        self.quanlity_weight = quality_weight
        self.diversity_weight = diversity_weight

        self.reset()
    
    def reset(self) -> State:
        """Reset environment for new episode."""
        self.current_query = None
        self.available_communities = []
        self.selected_communities = []
        self.episode_rewards = []
        return self._get_current_state()
        
    def set_query_and_communities(self, 
                                 query_embedding: np.ndarray,
                                 communities: List[CommunityFeatures]):
        """Initialize environment with query and available communities."""
        self.current_query = query_embedding
        self.available_communities = communities
        self.selected_communities = []
        self.episode_rewards = []

    def step(self , action_idx : int  ) -> Tuple[State , torch.FloatTensor, bool , Dict[str , Any]]:
        """returns new state , action , done flag and info"""

        if action_idx >= len(self.available_communities):
            reward = -1.0 
            done = True
            info = {"error" : "invalid action"}
            return self._get_current_state() , reward , done , info 
        

        community = self.available_communities[action_idx]

        if community in self.selected_communities:
            reward = 0.5 
            done = False 
            info = {"warning" : "community selected"}
        
        else:
            self.selected_communities.append(community.community_id)
            reward = self.calculate_reward(community)
            info = {
                "selected community" : community.community_id,
                "reward" : self.get_reward_components(community)
            }

        done = len(self.selected_communities) >= self.max_selections
        
        return self._get_current_state(), reward, done, info
    
    def _get_current_state(self) -> State :
        return State(
            query_embedding= self.current_query if self.current_query is not None else np.zeros(384),
            community_features= self.available_communities, 
            selected_communities= self.selected_communities,
            remaining_budget= self.max_selections - len(self.selected_communities)
        )
    
    def _calculate_reward(self , community : CommunityFeatures) -> torch.FloatTensor: 
        components = self._get_reward_components(community)
        component_keys = components.keys

        if component_keys == "relevance" or "quality" or "diversity": 
            reward = self.relevance_weight*components['relevance'] + self.diversity_weight*components['diversity'] + self.quality_weight * components['quality']
            
            reward_tensor = torch.FloatTensor(reward , device = torch.device)

        return reward_tensor
            
    def _get_reward_components(self , community: CommunityFeatures) -> Dict[str , float]:


        query_similarity = cosine_similarity(
            self.current_query.reshape(-1 , 1) , community.feature_vector[:384].reshape(-1 , 1)
        )[0,0]

        quality_score = community.confidence_score

        diversity_score = 1.0

        if self.selected_communities:
            selected_features = np.array([
                cf.feature_vector for cf in self.available_communities
                if cf.community_id in self.selected_communities
            ])
            
            if selected_features.shape[0] > 0:
                diversity_similarities = cosine_similarity(
                    community.feature_vector.reshape(1, -1),
                    selected_features
                )
                diversity_score = 1.0 - np.mean(diversity_similarities)


        return {
            'relevance': float(query_similarity),
            'quality': float(quality_score),
            'diversity': float(diversity_score)
        }
    
    

@dataclass
class Reward: 
    """Reward for the action. It will depend on probable a certain community is chosen from a 
    bunch of communities"""

    