"""
Deep Q-Network (DQN) Model for Pacman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """Residual block with 3D convolutions for temporal sequences."""
    
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class DQN(nn.Module):
    """
    Deep Q-Network with 3D Convolutional Backbone.
    
    Architecture:
    - Input: (batch_size, 5, time, height, width) - 5 channels over time
    - Backbone: 3D Convs + Residual Blocks
    - Output: (batch_size, 4) - Q-values for each action
    """
    
    def __init__(self, input_channels=5, conv_channels=64, num_residual_blocks=3, num_actions=4):
        super(DQN, self).__init__()
        
        # ============ Shared Backbone (3D Convolutions) ============
        # Input: (batch, channels, time, height, width)
        self.conv_input = nn.Conv3d(input_channels, conv_channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.bn_input = nn.BatchNorm3d(conv_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock3D(conv_channels) for _ in range(num_residual_blocks)
        ])
        
        # Additional conv layer before temporal pooling
        self.conv_output = nn.Conv3d(conv_channels, 32, kernel_size=(3,3,3), padding=(1,1,1))
        self.bn_output = nn.BatchNorm3d(32)
        
        # Flattened size calculation will happen dynamically
        self.fc_shared = None
        self.fc_shared_input_size = None
        self.fc_features_size = 256
        
        # ============ Q-Value Head ============
        self.fc1 = nn.Linear(self.fc_features_size, 128)
        self.fc2 = nn.Linear(128, num_actions)
        
    def forward_backbone(self, x):
        """Extract features using 3D convolutions."""
        # Initial 3D convolution
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Output 3D convolution
        out = F.relu(self.bn_output(self.conv_output(out)))
        
        # Temporal pooling: collapse time dimension
        out = F.adaptive_avg_pool3d(out, (1, out.size(3), out.size(4)))
        out = out.squeeze(2)  # Remove time dimension -> (batch, channels, height, width)
        
        # Flatten spatial dimensions
        out = out.view(out.size(0), -1)
        
        # Initialize fc_shared on first pass if needed
        if self.fc_shared is None or self.fc_shared_input_size != out.size(1):
            self.fc_shared_input_size = out.size(1)
            self.fc_shared = nn.Linear(out.size(1), self.fc_features_size).to(out.device)
        
        # Fully connected layer to get feature vector
        out = F.relu(self.fc_shared(out))
        return out
        
    def forward(self, x):
        """
        Forward pass.
        Returns: Q-values (batch, 4)
        """
        features = self.forward_backbone(x)
        
        out = F.relu(self.fc1(features))
        q_values = self.fc2(out)
        
        return q_values
