"""
Deep Q-Network (DQN) Model for Pacman
Optimized with 2D Convolutions (Time stacked on Channel dimension)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network with 2D Convolutional Backbone.
    
    Architecture:
    - Input: (batch_size, input_channels * frames, height, width) 
      -> 5 channels * 5 frames = 25 channels
    - Backbone: 2D Convs
    - Output: (batch_size, 4) - Q-values for each action
    """
    
    def __init__(self, input_channels=5, num_frames=5, conv_channels=32, num_actions=4):
        super(DQN, self).__init__()
        
        self.input_dim = input_channels * num_frames # 25
        
        # simple CNN architecture similar to Atari DQN nature paper but scaled down for Pacman
        self.conv1 = nn.Conv2d(self.input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Flattened size calculation will happen dynamically
        self.fc_shared = None
        self.fc_input_size = None
        
        # ============ Q-Value Head ============
        self.fc1 = nn.Linear(512, 128) # Input size will be determined dynamically, mapping to 512 first
        self.fc2 = nn.Linear(128, num_actions)
        
    def forward(self, x):
        """
        Forward pass.
        x: (batch, 25, H, W)
        """
        # Convolutional Layers
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Dynamic FC layer initialization
        if self.fc_shared is None or self.fc_input_size != out.size(1):
            self.fc_input_size = out.size(1)
            self.fc_shared = nn.Linear(self.fc_input_size, 512).to(out.device)
            
        out = F.relu(self.fc_shared(out))
        
        out = F.relu(self.fc1(out))
        q_values = self.fc2(out)
        
        return q_values
