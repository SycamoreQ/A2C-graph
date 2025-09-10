import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from environment import State , Action , Reward , Environment
from typing import Dict , Any , List , Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    
    def __init__(self ,state_dim:int ,  max_communities:int , hidden_dim = 256):
        super.__init__()
        
        self.max_communities = max_communities
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layer: probability for each possible community
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_communities)
        )


    def forward(self , state: torch.Tensor):
        state_features = self.state_encoder(state)
        policy_logits = self.policy_head(state_features)
        action_probs = F.softmax(policy_logits, dim=-1)

        return action_probs
    

    def select_action(self, state: State) -> Tuple[int, torch.Tensor]:
        """
        Select an action (community) given the current state
        
        Returns:
            action_idx: Index of selected community
            log_prob: Log probability of selected action (for training)
        """
        state_tensor = state.to_flat_tensor().unsqueeze(0)
        
        action_probs = self.forward(state_tensor)
        
        valid_mask = torch.tensor([
            0.0 if comm_id in state.selected_communities else 1.0 
            for comm_id in state.available_communities
        ])
        
        masked_probs = action_probs.squeeze() * valid_mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs = valid_mask / valid_mask.sum()
        
        dist = Categorical(masked_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
