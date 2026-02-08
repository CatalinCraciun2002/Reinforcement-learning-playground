"""
DQN Agent for Pacman
Implements the Agent interface and handles Replay Buffer and Epsilon-Greedy action selection.
"""
import torch
import numpy as np
import random
from collections import deque
import sys
import os

# Add parent directory to path to import game module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.game import Agent, Directions
from models.deep_qlearning_models.dqn_model import DQN

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent(Agent):
    """
    DQN Agent that uses epsilon-greedy exploration and a 2D ConvNet (stacked frames).
    """
    
    def __init__(self, index=0, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        super().__init__(index)
        
        # Frame buffer for temporal memory
        self.num_frames = 50
        
        self.model = DQN(num_frames=self.num_frames)
        self.target_model = DQN(num_frames=self.num_frames)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 320
        
        # Mapping actions to integers
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
        self.frame_buffer = deque(maxlen=self.num_frames)

    def state_to_tensor(self, state):
        """
        Convert game state to 5-channel tensor using optimized numpy operations.
        Output: (5, H, W) numpy array (float32)
        """
        walls_grid = state.getWalls()
        width, height = walls_grid.width, walls_grid.height
        
        # grids in pacman are [x][y] aka [col][row].
        # We need [row][col] for pytorch/numpy usually (H, W).
        
        # 1. Walls
        # walls_grid.data is list of lists [width][height]
        # np.array(walls_grid.data) -> shape (W, H)
        # Transpose to (H, W)
        walls_arr = np.array(walls_grid.data, dtype=np.float32).T
        
        # 2. Food
        food_grid = state.getFood()
        food_arr = np.array(food_grid.data, dtype=np.float32).T
        
        # 3. Pacman
        pacman_arr = np.zeros((height, width), dtype=np.float32)
        pacman_pos = state.getPacmanPosition()
        if pacman_pos:
            px, py = int(pacman_pos[0]), int(pacman_pos[1])
            # Check bounds just in case
            if 0 <= px < width and 0 <= py < height:
                # Note: arr is (y, x)
                pacman_arr[py, px] = 1.0
                
        # 4. Ghosts (Channel 1: Normal, Channel 3: Scared)
        ghosts_arr = np.zeros((height, width), dtype=np.float32)
        scared_ghosts_arr = np.zeros((height, width), dtype=np.float32)
        
        for ghost_state in state.getGhostStates():
            pos = ghost_state.getPosition()
            if pos:
                gx, gy = int(pos[0]), int(pos[1])
                if 0 <= gx < width and 0 <= gy < height:
                    if ghost_state.scaredTimer > 0:
                        scared_ghosts_arr[gy, gx] = 1.0
                    else:
                        ghosts_arr[gy, gx] = 1.0
        
        # Stack channels: (Pacman, Ghosts, Walls, ScaredGhosts, Food)
        # Consistent with previous ordering: 0:Pac, 1:Ghost, 2:Wall, 3:Scared, 4:Food
        channels = np.stack([
            pacman_arr,
            ghosts_arr,
            walls_arr,
            scared_ghosts_arr,
            food_arr
        ], axis=0) # (5, H, W)
        
        return torch.from_numpy(channels) # Return tensor CPU

    def registerInitialState(self, state):
        initial_tensor = self.state_to_tensor(state)
        self.frame_buffer.clear()
        for _ in range(self.num_frames):
            self.frame_buffer.append(initial_tensor)
            
    def get_temporal_input(self, state):
        """
        Return stacked frames tensor.
        Instead of (1, 5, 5, H, W), we flat stack to (1, 25, H, W)
        """
        current_tensor = self.state_to_tensor(state)
        self.frame_buffer.append(current_tensor)
        
        # Concatenate along channel dim (dim 0 for single tensors, but we want (1, C, H, W))
        # stored items are (5, H, W)
        # Stack -> (5, 5, H, W) -> Flatten to (25, H, W)
        # Or just cat dim=0
        
        stacked = torch.cat(list(self.frame_buffer), dim=0) # (25, H, W)
        return stacked.unsqueeze(0) # (1, 25, H, W)

    def getAction(self, state):
        """
        Epsilon-greedy action selection.
        """
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
        if not legal:
            return Directions.STOP
            
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.choice(legal)
        
        # Model Inference
        state_tensor = self.get_temporal_input(state) # (1, 25, H, W)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze() # (4,)
            
        # Mask illegal actions with -inf
        action_scores = []
        for i, action in enumerate(self.actions):
            if action in legal:
                action_scores.append((q_values[i].item(), action))
            else:
                action_scores.append((-float('inf'), action))
                
        # Return action with max Q-value
        best_action = max(action_scores, key=lambda x: x[0])[1]
        return best_action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
