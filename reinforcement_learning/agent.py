"""
Agent for Pacman Reinforcement Learning

The agent uses a unified Actor-Critic network to predict action probabilities
and value estimates for a given game state.
Input: 5-channel grid (pacman, ghost, wall, scared ghost, food)
Output: 4 action probabilities (North, South, East, West)
"""

import torch
import numpy as np
import sys
import os
from collections import deque

# Add parent directory to path to import game module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.game import Agent, Directions
from reinforcement_learning.model import ActorCriticNetwork


class RLAgent(Agent):
    """RL Agent with position-based memory."""
    
    def __init__(self, index=0, memory_context=5):
        super().__init__(index)
        self.model = ActorCriticNetwork(memory_context=memory_context)
        self.model.eval()
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        self.memory_context = memory_context
        self.position_buffer = deque(maxlen=memory_context)
    
    def state_to_tensor(self, state):
        """
        Convert game state to 5-channel tensor.
        
        Channels:
        0: Pacman position
        1: Ghost positions
        2: Walls
        3: Scared ghosts
        4: Food
        
        Returns:
            torch.Tensor: Shape (1, 5, height, width)
        """
        # Get grid dimensions
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # 5 base channels
        channels = np.zeros((5, height, width), dtype=np.float32)
        
        # Channel 0: Pacman
        x, y = int(state.getPacmanPosition()[0]), int(state.getPacmanPosition()[1])
        channels[0, y, x] = 1.0
        
        # Channels 1 & 3: Ghosts
        for ghost in state.getGhostStates():
            x, y = int(ghost.getPosition()[0]), int(ghost.getPosition()[1])
            channels[3 if ghost.scaredTimer > 0 else 1, y, x] = 1.0
        
        # Channel 2: Walls (vectorized)
        channels[2] = np.array([[walls[x][y] for y in range(height)] for x in range(width)], dtype=np.float32).T
        
        # Channel 4: Food (vectorized)
        food = state.getFood()
        channels[4] = np.array([[food[x][y] for y in range(height)] for x in range(width)], dtype=np.float32).T
        
        # Append position history channels
        for pos_channel in self.position_buffer:
            channels = np.concatenate([channels, pos_channel], axis=0)
        
        return torch.from_numpy(channels).unsqueeze(0)
    
    def registerInitialState(self, state):
        """Initialize position buffer with initial Pacman position."""
        walls = state.getWalls()
        width, height = walls.width, walls.height
        x, y = int(state.getPacmanPosition()[0]), int(state.getPacmanPosition()[1])
        
        initial_pos = np.zeros((1, height, width), dtype=np.float32)
        initial_pos[0, y, x] = 1.0
        
        self.position_buffer = deque([initial_pos] * self.memory_context, maxlen=self.memory_context)
    
    def update_position_buffer(self, state):
        """Add current Pacman position to buffer."""
        walls = state.getWalls()
        width, height = walls.width, walls.height
        x, y = int(state.getPacmanPosition()[0]), int(state.getPacmanPosition()[1])
        
        pos_channel = np.zeros((1, height, width), dtype=np.float32)
        pos_channel[0, y, x] = 1.0
        self.position_buffer.append(pos_channel)
    
    def getAction(self, state):
        """Select action based on current game state."""
        legal_actions = state.getLegalPacmanActions()
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        if not legal_actions:
            return Directions.STOP
        
        # Update position buffer before getting state tensor
        self.update_position_buffer(state)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = self.model(self.state_to_tensor(state), return_both=False).squeeze().numpy()
        
        # Mask and normalize
        mask = np.array([1.0 if a in legal_actions else 0.0 for a in self.actions])
        probs = action_probs * mask
        probs = probs / probs.sum() if probs.sum() > 0 else mask / mask.sum()
        
        return self.actions[np.random.choice(len(self.actions), p=probs)]
