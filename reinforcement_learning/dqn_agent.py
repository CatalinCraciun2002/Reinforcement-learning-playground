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
from game import Agent, Directions
from reinforcement_learning.dqn_model import DQN

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
    DQN Agent that uses epsilon-greedy exploration and a 3D ConvNet.
    """
    
    def __init__(self, index=0, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        super().__init__(index)
        
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        
        # Mapping actions to integers
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
        # Frame buffer for temporal memory (stores last 5 frames)
        self.frame_buffer = deque(maxlen=5)

    def state_to_tensor(self, state):
        """
        Convert game state to 5-channel tensor.
        SAME LOGIC AS RLAgent.
        """
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        channels = np.zeros((5, height, width), dtype=np.float32)
        
        # Channel 0: Pacman
        pacman_pos = state.getPacmanPosition()
        if pacman_pos:
            x, y = int(pacman_pos[0]), int(pacman_pos[1])
            if 0 <= x < width and 0 <= y < height:
                channels[0, y, x] = 1.0
        
        # Ghosts
        ghost_states = state.getGhostStates()
        for ghost_state in ghost_states:
            pos = ghost_state.getPosition()
            if pos:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < width and 0 <= y < height:
                    if ghost_state.scaredTimer > 0:
                        channels[3, y, x] = 1.0
                    else:
                        channels[1, y, x] = 1.0
        
        # Walls
        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    channels[2, y, x] = 1.0
                    
        # Food
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    channels[4, y, x] = 1.0
        
        return torch.from_numpy(channels).unsqueeze(0) # (1, 5, H, W)

    def registerInitialState(self, state):
        initial_tensor = self.state_to_tensor(state)
        self.frame_buffer.clear()
        for _ in range(5):
            self.frame_buffer.append(initial_tensor)
            
    def get_temporal_input(self, state):
        """Return stacked frames tensor (1, 5, 5, H, W)."""
        current_tensor = self.state_to_tensor(state)
        self.frame_buffer.append(current_tensor)
        stacked = torch.stack(list(self.frame_buffer), dim=2)
        return stacked

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
        state_tensor = self.get_temporal_input(state) # (1, 5, 5, H, W)
        
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
