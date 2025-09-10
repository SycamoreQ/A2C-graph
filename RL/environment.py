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
                                 query_embedding: np.ndarray):
        """Initialize environment with query and available communities."""
        self.current_query = query_embedding

        self.available_communities = State.available_communities
        self.selected_communities = []
        self.episode_rewards = []

    def step(self , action_idx : int) -> Tuple[State , torch.FloatTensor, bool , Dict[str , Any]]:
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

@dataclass
class Agent:

    def __init__(self , state_dim: int , max_communities: int = 30  , device: str = 'gpu'):
        self.device = torch.device(device)

        self.actor = Actor(state_dim , max_communities).to(self.device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.gamma = 0.99 
        self.entropy_coeff = 0.01 
    
    def select_communities(self , init_state: State , max_selection:int = 5) -> Tuple[List[str] , Dict[str, Any]]: 
        current_state = init_state
        
        action_info = {
            'actions_taken' : [],
            'action_prob' : [],
            'state_values' : [],
            'rewards_received' : []
        }

        selected_communities = []

        for _ in range(max_selection): 
            action_idx , action_probs = self.actor.select_action(current_state)
            state_value = self.critic.evaluate_state(current_state)

            if action_idx >= len(current_state.available_communities):
                break 

            selected_community = current_state.available_communities[action_idx]
            selected_communities.append(selected_community)

            action_info['actions_taken'].append(action_idx)
            action_info['action_probs'].append(action_probs)
            action_info['state_values'].append(state_value)
            
            current_state.selected_communities.append(selected_community)
            current_state.remaining_budget -= 1
            
            reward = Environment._calculate_reward()
            action_info['rewards_received'].append(reward)
            
            if current_state.remaining_budget <= 0:
                break
        
        return selected_communities, action_info
    


@dataclass
class Reward:
    weights: Dict[str , float]
    community_tensor =  State.llm_community_features
    query = State.llm_query_features
    selected = State.selected_communities
    available = State.available_communities
    threshold: float

    def diversity_score(self , diversity_weight: int = 0.2):
        """Diversity check to see if model doesnt run in loop"""

        if self.selected:
            similarity = cosine_similarity(self.available.flatten() , self.selected.flatten())
        else:
            similarity = 1.0 

        diversity = torch.FloatTensor(diversity_weight)*(1 - np.mean(similarity))
        return diversity
        
    def _calculate_reward(self) -> torch.Tensor:
        
        query_score = cosine_similarity(self.query.flatten() , self.community_tensor.flatten()) 
        
        if query_score > self.threshold:
            total_reward = 0.5

        
    



        