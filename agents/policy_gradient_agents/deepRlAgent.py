"""
Agent for Pacman Reinforcement Learning

The agent uses a unified Actor-Critic network to predict action probabilities
and value estimates for a given game state.
Input: 6-channel grid (pacman, ghost, wall, scared ghost, food, capsules)
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
from models.policy_gradient_models.simple_residual_conv import ActorCriticNetwork


class RLAgent(Agent):
    """RL Agent with position-based memory."""
    
    def __init__(self, model, memory_context=5, device=None):

        self.model = model
        self.device = device if device else next(model.parameters()).device
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.memory_context = memory_context
        self.position_buffers = {}  # env_id -> deque for batched processing
        self.wall_cache = {}  # env_id -> cached wall array (walls never change)

    
    def state_to_tensor(self, state, env_id=0):
        """Convert game state to 6-channel tensor.
        
        Channels:
        0: Pacman position
        1: Ghost positions
        2: Walls
        3: Scared ghosts
        4: Food (regular pellets only)
        5: Capsules (power pellets)
        
        Args:
            state: Game state
            env_id: Environment ID for retrieving correct position buffer
        
        Returns:
            torch.Tensor: Shape (1, 6+memory_context, height, width)
        """
        # Get grid dimensions
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        buffer = self.position_buffers.get(env_id, [])
        num_pos_channels = len(buffer)
        
        # 6 base channels + memory context channels
        channels = np.zeros((6 + num_pos_channels, height, width), dtype=np.float32)
        
        # Channel 0: Pacman
        px, py = state.getPacmanPosition()
        channels[0, int(py), int(px)] = 1.0
        
        # Channels 1 & 3: Ghosts
        for ghost in state.getGhostStates():
            gx, gy = ghost.getPosition()
            channels[3 if ghost.scaredTimer > 0 else 1, int(gy), int(gx)] = 1.0
        
        # Channel 2: Walls (cached - walls never change)
        if env_id not in self.wall_cache:
            # Direct conversion from Grid.data (list of lists)
            self.wall_cache[env_id] = np.array(walls.data, dtype=np.float32).T
        channels[2] = self.wall_cache[env_id]
        
        # Channel 4: Food (optimized with Grid.data)
        food = state.getFood()
        capsules = state.getCapsules()
        # Direct conversion from Grid.data - much faster than nested comprehension
        food_array = np.array(food.data, dtype=np.float32).T
        
        # Vectorized capsule removal
        if capsules:
            for cx, cy in capsules:
                food_array[int(cy), int(cx)] = 0.0
        channels[4] = food_array
        
        # Channel 5: Capsules (power pellets)
        for cx, cy in capsules:
            channels[5, int(cy), int(cx)] = 1.0
        
        # Append position history channels from environment-specific buffer
        for i, (hx, hy) in enumerate(buffer):
            channels[6 + i, hy, hx] = 1.0
        
        return torch.from_numpy(channels).unsqueeze(0)
    
    def registerInitialState(self, state, env_id=0):
        """Initialize position buffer with initial Pacman position.
        
        Args:
            state: Initial game state
            env_id: Environment ID for buffer tracking
        """
        x, y = int(state.getPacmanPosition()[0]), int(state.getPacmanPosition()[1])
        
        # Fill all buffer slots with initial position (stored as tuples)
        self.position_buffers[env_id] = deque([(x, y)] * self.memory_context, maxlen=self.memory_context)
    
    def update_position_buffer(self, state, env_id=0):
        """Add current Pacman position to buffer.
        
        Args:
            state: Current game state
            env_id: Environment ID for buffer tracking
        """
        x, y = int(state.getPacmanPosition()[0]), int(state.getPacmanPosition()[1])
        
        self.position_buffers[env_id].append((x, y))
    
    def get_action_mask(self, legal_actions):
        """Creates a binary mask tensor for legal actions."""
        return torch.tensor([1.0 if a in legal_actions else 0.0 for a in self.actions], dtype=torch.float32)

    def getAction(self, legal_actions, action_probs):

        mask = self.get_action_mask(legal_actions).to(action_probs.device)
        masked = action_probs * mask
        if masked.sum() > 0:
            masked = masked / masked.sum()
        else:
            # Fallback if no probability mass on legal acts (shouldn't happen with proper masking)
            masked = mask / mask.sum()
        action_idx = torch.multinomial(masked, 1).item()

        return self.actions[action_idx], action_idx

    def forward(self, state, env_id=0):
        """Forward pass for a single state.
        
        Args:
            state: Game state
            env_id: Environment ID for buffer tracking
            
        Returns:
            probs: Action probabilities (5,)
            value: State value estimate (scalar)
        """
        state_tensor = self.state_to_tensor(state, env_id)
        probs, value = self.model(state_tensor, return_both=True)
        probs = probs.squeeze()
        value = value.squeeze()

        self.update_position_buffer(state, env_id)

        return probs, value
    
    def forward_batch(self, states, env_ids, action_masks=None):
        """Batched forward pass for multiple states.
        
        Args:
            states: List of game states
            env_ids: List of environment IDs corresponding to states
            action_masks: Optional list of action mask tensors
            
        Returns:
            probs_batch: Action probabilities (batch_size, 5)
            values_batch: State value estimates (batch_size,)
        """
        
        # Convert all states to tensors and stack into batch
        state_tensors = [self.state_to_tensor(state, env_id) for state, env_id in zip(states, env_ids)]
        batch_tensor = torch.cat(state_tensors, dim=0).to(self.device)  # (batch_size, channels, H, W)
        
        if action_masks is not None:
            action_masks = torch.stack(action_masks).to(self.device)
        
        # Single batched forward pass
        probs_batch, values_batch = self.model(batch_tensor, return_both=True, action_mask=action_masks)
        values_batch = values_batch.squeeze(-1)  # (batch_size,)

        # Update position buffers for all states
        for state, env_id in zip(states, env_ids):
            self.update_position_buffer(state, env_id)
        
        return probs_batch, values_batch
