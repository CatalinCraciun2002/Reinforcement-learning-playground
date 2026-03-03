"""
Actor-Critic Multitask Model for Pacman Reinforcement Learning

This model combines actor (policy) and critic (value) networks with shared backbone.
Uses position-based memory: appends Pacman position history as additional channels.

Input: (6 + memory_context) channels
  - 6 channels: current state (pacman, ghosts, walls, scared ghosts, food, capsules)
  - memory_context channels: Pacman positions from previous timesteps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with 2D convolutions."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic multitask model with position-based memory.
    
    Args:
        memory_context: Number of previous Pacman positions to track (default: 5)
        conv_channels: Number of convolutional channels (default: 64)
        num_residual_blocks: Number of residual blocks (default: 3)
    """
    
    def __init__(self, memory_context=5, conv_channels=64, num_residual_blocks=3, 
                 spatial_height=11, spatial_width=20):
        """
        Initialize Actor-Critic network.
        
        Args:
            memory_context: Number of previous Pacman positions to track
            conv_channels: Number of convolutional channels
            num_residual_blocks: Number of residual blocks
            spatial_height: Height of the input grid (default: 11 for mediumClassic)
            spatial_width: Width of the input grid (default: 20 for mediumClassic)
        """
        super().__init__()
        
        input_channels = 6 + memory_context  # Current state + position history
        
        # Shared backbone
        self.conv_input = nn.Conv2d(input_channels, conv_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(conv_channels)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(conv_channels) for _ in range(num_residual_blocks)
        ])
        
        self.conv_output = nn.Conv2d(conv_channels, 32, 3, padding=1)
        self.bn_output = nn.BatchNorm2d(32)
        
        # Static FC layer - size determined by spatial dimensions
        # Since we use same-padding convolutions, spatial dimensions are preserved
        fc_input_size = 32 * spatial_height * spatial_width
        self.fc_shared = nn.Linear(fc_input_size, 256)
        
        # Actor head
        self.actor_fc1 = nn.Linear(256, 128)
        self.actor_fc2 = nn.Linear(128, 5)  # 5 actions: North, South, East, West, Stop
        
        # Critic head
        self.critic_fc1 = nn.Linear(256, 128)
        self.critic_fc2 = nn.Linear(128, 1)
        
    def forward_backbone(self, x):
        """Shared feature extraction."""
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.residual_blocks:
            out = block(out)
        
        out = F.relu(self.bn_output(self.conv_output(out)))
        out = out.view(out.size(0), -1)
        
        return F.relu(self.fc_shared(out))
    
    def forward_actor(self, shared_features):
        """Policy head: action probabilities and log-probabilities."""
        out = F.relu(self.actor_fc1(shared_features))
        log_probs = F.log_softmax(self.actor_fc2(out), dim=1)
        return log_probs.exp(), log_probs  # (probs, log_probs)
    
    def forward_critic(self, shared_features):
        """Value head: state value estimate."""
        out = F.relu(self.critic_fc1(shared_features))
        return self.critic_fc2(out)
    
    def forward(self, x, return_both=True):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 5+memory_context, H, W)
            return_both: Return both actor and critic outputs
            
        Returns:
            (action_probs, values) if return_both else action_probs
        """
        features = self.forward_backbone(x)
        probs, self.last_log_probs = self.forward_actor(features)

        if return_both:
            return probs, self.forward_critic(features)
        return probs
