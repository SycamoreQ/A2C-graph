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



@dataclass
class State:
    """State representation for the RL environment."""
    query_embedding: np.ndarray
    community_features: List[CommunityFeatures] 
    selected_communities: List[str]
    remaining_budget: int
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        community_matrix = np.stack([cf.feature_vector for cf in self.community_features])
        
        selection_mask = np.array([
            1.0 if cf.community_id in self.selected_communities else 0.0 
            for cf in self.community_features
        ])
        
        community_agg = np.mean(community_matrix, axis=0)
        
        state_vector = np.concatenate([
            self.query_embedding,
            community_agg,
            selection_mask,
            [self.remaining_budget / 10.0]  
        ])
        
        return torch.FloatTensor(state_vector).to(device)
    

@dataclass
class ActionSpace: 
    action_index: int 
    action: List[Action] 
    space_size: List[int , int]
    
    def to_tensor(self) -> torch.Tensor:

        action_space_tensor = torch.zeros(self.space_size)

        action_space_tensor[action_space_tensor[:,0] , action_space_tensor[:,1]] = torch.tensor(self.action).float()


        return action_space_tensor  

        

        

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

    