import torch 
from torch.nn import nn
from .environment import State , Action , Reward , Environment 
from .A2C import compute_returns
from typing import List , Any , Dict , Tuple , Optional
from collections import deque
import logging 
import torch.optim as optim 
import numpy as np 
from Graph.data.community import CommunityFeatures
import torch.functional as F
from torch.distributions import Categorical

class A2CTrainer:
    """A2C trainer for community selection."""
    
    def __init__(self,
                 model: compute_returns,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5,
                 device: torch.device = torch.device('cpu')):
        
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.device = device
        
        self.training_rewards = deque(maxlen=1000)
        self.training_losses = deque(maxlen=1000)

    def train_episode(self, 
                     env: Environment,
                     query_embedding: np.ndarray,
                     communities: List[CommunityFeatures]) -> Dict[str, float]:
        """Train on a single episode with full tensor operations."""
        
        env.set_query_and_communities(query_embedding, communities)
        state = env.reset()
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        done = False
        total_reward = torch.tensor(0.0, device=self.device)
        
        while not done:
            state_tensor = state.to_flat_tensor().unsqueeze(0)
            
            action_probs, state_value = self.model(state_tensor)
            
            # Mask invalid actions
            valid_mask = state.get_valid_action_mask()
            masked_probs = action_probs * valid_mask.float()
            
            # Renormalize probabilities
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # If all actions are invalid, uniform over valid actions
                masked_probs = valid_mask.float() / valid_mask.float().sum()
            
            # Sample action
            dist = Categorical(masked_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store trajectory
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(state_value)
            
            # Take action in environment (returns tensors)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            total_reward += reward
        
        # Convert lists to tensors
        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.cat(values)
        log_probs_tensor = torch.stack(log_probs)
        
        # Calculate returns and advantages
        returns = self._calculate_tensor_returns(rewards_tensor)
        advantages = self._calculate_tensor_advantages(returns, values_tensor)
        
        # Compute losses
        policy_loss = self._compute_tensor_policy_loss(log_probs_tensor, advantages)
        value_loss = self._compute_tensor_value_loss(values_tensor, returns)
        entropy_loss = self._compute_tensor_entropy_loss(log_probs_tensor)
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_loss_coeff * value_loss - 
            self.entropy_coeff * entropy_loss
        )
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Store statistics
        self.training_rewards.append(total_reward.item())
        self.training_losses.append(total_loss.item())
        
        return {
            'total_reward': total_reward.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'selected_communities': [
                state.community_metadata[i].community_id 
                for i, selected in enumerate(state.selected_mask) 
                if selected.item() == 1.0
            ]
        }
    
    def _calculate_tensor_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns using tensors."""
        returns = torch.zeros_like(rewards)
        R = torch.tensor(0.0, device=self.device)
        
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
            
        return returns
    
    def _calculate_tensor_advantages(self, 
                                   returns: torch.Tensor, 
                                   values: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using tensor operations."""
        advantages = returns - values.squeeze()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def _compute_tensor_policy_loss(self, 
                                  log_probs: torch.Tensor,
                                  advantages: torch.Tensor) -> torch.Tensor:
        """Compute policy loss using tensor operations."""
        return -(log_probs * advantages.detach()).mean()
    
    def _compute_tensor_value_loss(self, 
                                 values: torch.Tensor,
                                 returns: torch.Tensor) -> torch.Tensor:
        """Compute value function loss using tensors."""
        return F.mse_loss(values.squeeze(), returns)
    
    def _compute_tensor_entropy_loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy loss using tensors."""
        return log_probs.mean()  # Negative entropy for regularization
    
    def train_batch(self, 
                   episodes_data: List[Tuple[np.ndarray, List[CommunityFeatures]]]) -> Dict[str, float]:
        """Train on a batch of episodes for improved efficiency."""
        
        batch_size = len(episodes_data)
        batch_losses = []
        batch_rewards = []
        
        for query_embedding, communities in episodes_data:
            episode_result = self.train_episode(
                self.environment, query_embedding, communities
            )
            batch_losses.append(episode_result['total_loss'])
            batch_rewards.append(episode_result['total_reward'])
        
        return {
            'batch_avg_loss': np.mean(batch_losses),
            'batch_avg_reward': np.mean(batch_rewards),
            'batch_reward_std': np.std(batch_rewards),
            'episodes_processed': batch_size
        } 
    
    def _calculate_tensor_returns(self , rewards: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        R = torch.tensor(0.0, device=self.device)

        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
            
        return torch.FloatTensor(returns).to(self.device)
    
    def _calculate_advantages(self, 
                            returns: torch.Tensor, 
                            values: List[torch.Tensor]) -> torch.Tensor:
        """Calculate advantages using returns and value estimates."""
        values_tensor = torch.cat(values).squeeze()
        advantages = returns - values_tensor
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def _compute_policy_loss(self, 
                           log_probs: List[torch.Tensor],
                           advantages: torch.Tensor) -> torch.Tensor:
        """Compute policy loss."""
        policy_losses = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_losses.append(-log_prob * advantage)
        return torch.stack(policy_losses).mean()
    
    def _compute_value_loss(self, 
                          values: List[torch.Tensor],
                          returns: torch.Tensor) -> torch.Tensor:
        """Compute value function loss."""
        values_tensor = torch.cat(values).squeeze()
        return F.mse_loss(values_tensor, returns)
    
    def _compute_entropy_loss(self, log_probs: List[torch.Tensor]) -> torch.Tensor:
        """Compute entropy loss for exploration."""
        entropies = [-log_prob for log_prob in log_probs]
        return torch.stack(entropies).mean()
    
    def get_training_statistics(self) -> Dict[str, float]:
        """Get current training statistics."""
        return {
            'avg_reward': np.mean(self.training_rewards) if self.training_rewards else 0.0,
            'avg_loss': np.mean(self.training_losses) if self.training_losses else 0.0,
            'reward_std': np.std(self.training_rewards) if len(self.training_rewards) > 1 else 0.0
        }