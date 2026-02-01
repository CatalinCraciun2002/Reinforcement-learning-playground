"""
Data Loader for Human Gameplay Recordings

Utilities to load and process recorded human gameplay for use in training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import glob
import numpy as np
from typing import List, Dict, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features will be disabled.")


class GameplayDataset:
    """Dataset of human gameplay recordings."""
    
    def __init__(self, data_dir='game_runs_data'):
        """Load all gameplay recordings from the specified directory."""
        # Get absolute path to data directory
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                data_dir
            )
        
        self.data_dir = data_dir
        self.episodes = []
        self.transitions = []
        
        self._load_all_episodes()
    
    def _load_all_episodes(self):
        """Load all pickle files from the data directory."""
        pattern = os.path.join(self.data_dir, '*.pkl')
        pkl_files = glob.glob(pattern)
        
        if not pkl_files:
            print(f"Warning: No gameplay recordings found in {self.data_dir}")
            return
        
        print(f"Loading {len(pkl_files)} gameplay recording(s)...")
        
        for pkl_path in pkl_files:
            try:
                with open(pkl_path, 'rb') as f:
                    episode_data = pickle.load(f)
                    self.episodes.append(episode_data)
                    self.transitions.extend(episode_data['transitions'])
                    
                    outcome = episode_data['outcome']
                    score = episode_data['final_score']
                    steps = episode_data['num_steps']
                    print(f"  Loaded: {os.path.basename(pkl_path)} "
                          f"({outcome}, Score: {score}, Steps: {steps})")
                    
            except Exception as e:
                print(f"Error loading {pkl_path}: {e}")
        
        print(f"\nTotal episodes: {len(self.episodes)}")
        print(f"Total transitions: {len(self.transitions)}")
    
    def get_statistics(self):
        """Get statistics about the loaded gameplay data."""
        if not self.episodes:
            return "No episodes loaded"
        
        wins = sum(1 for ep in self.episodes if ep['outcome'] == 'WIN')
        losses = sum(1 for ep in self.episodes if ep['outcome'] == 'LOSS')
        scores = [ep['final_score'] for ep in self.episodes]
        steps = [ep['num_steps'] for ep in self.episodes]
        rewards = [sum(t['reward'] for t in ep['transitions']) for ep in self.episodes]
        
        stats = {
            'num_episodes': len(self.episodes),
            'num_transitions': len(self.transitions),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self.episodes) if self.episodes else 0,
            'avg_score': np.mean(scores),
            'avg_steps': np.mean(steps),
            'avg_total_reward': np.mean(rewards),
            'score_range': (min(scores), max(scores)),
        }
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        
        if isinstance(stats, str):
            print(stats)
            return
        
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Episodes:        {stats['num_episodes']}")
        print(f"Total Transitions: {stats['num_transitions']}")
        print(f"Wins:            {stats['wins']}")
        print(f"Losses:          {stats['losses']}")
        print(f"Win Rate:        {stats['win_rate']:.2%}")
        print(f"Avg Score:       {stats['avg_score']:.1f}")
        print(f"Score Range:     {stats['score_range'][0]:.0f} - {stats['score_range'][1]:.0f}")
        print(f"Avg Steps:       {stats['avg_steps']:.1f}")
        print(f"Avg Total Reward: {stats['avg_total_reward']:.1f}")
        print("="*60 + "\n")
    
    def state_to_tensor(self, state_dict, walls, past_positions=None, memory_length=5):
        """
        Convert state dictionary to tensor format (compatible with agent).
        
        Args:
            state_dict: Dictionary containing game state
            walls: Wall grid array
            past_positions: List of past Pacman positions (oldest to newest)
            memory_length: Number of past positions to include as memory channels
        
        Returns:
            Tensor with (5 + memory_length) channels: 
            [pacman, ghosts, walls, scared_ghosts, food, pos_history_1, ..., pos_history_N]
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for state_to_tensor. Please install torch.")
        
        food_grid = state_dict['food_grid']
        width, height = food_grid.shape[0], food_grid.shape[1]
        
        # Initialize 5 base channels (height, width)
        channels = np.zeros((5, height, width), dtype=np.float32)
        
        # Channel 0: Pacman position
        px, py = state_dict['pacman_pos']
        channels[0, int(py), int(px)] = 1
        
        # Channel 1: Ghost positions (non-scared)
        # Channel 3: Scared ghost positions
        for ghost_pos, scared_timer in zip(state_dict['ghost_positions'], 
                                           state_dict['ghost_scared_timers']):
            gx, gy = ghost_pos
            if scared_timer > 0:
                channels[3, int(gy), int(gx)] = 1  # Scared ghost
            else:
                channels[1, int(gy), int(gx)] = 1  # Normal ghost
        
        # Channel 2: Walls
        channels[2] = walls.T
        
        # Channel 4: Food
        channels[4] = food_grid.T
        
        # Add memory channels (past Pacman positions)
        if past_positions is None:
            past_positions = []
        
        # Pad with earliest position if we don't have enough history
        if len(past_positions) < memory_length:
            # If no past positions, use current position as padding
            pad_position = past_positions[0] if past_positions else state_dict['pacman_pos']
            past_positions = [pad_position] * (memory_length - len(past_positions)) + past_positions
        elif len(past_positions) > memory_length:
            # Take only the most recent memory_length positions
            past_positions = past_positions[-memory_length:]
        
        # Append position history channels
        for pos in past_positions:
            pos_channel = np.zeros((1, height, width), dtype=np.float32)
            ppx, ppy = pos
            pos_channel[0, int(ppy), int(ppx)] = 1.0
            channels = np.concatenate([channels, pos_channel], axis=0)
        
        return torch.FloatTensor(channels)
    
    def get_training_batch(self, batch_size=32, device='cpu', memory_length=5):
        """
        Get a random batch of transitions for training.
        
        Args:
            batch_size: Number of transitions to sample
            device: PyTorch device to load tensors to
            memory_length: Number of past Pacman positions to include
        
        Returns:
            states: Tensor of shape (batch_size, 5 + memory_length, height, width)
            actions: List of action strings
            rewards: Tensor of shape (batch_size,)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for get_training_batch. Please install torch.")
        
        if len(self.transitions) < batch_size:
            batch_size = len(self.transitions)
        
        # Sample random transitions
        indices = np.random.choice(len(self.transitions), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        
        for idx in indices:
            transition = self.transitions[idx]
            
            # Find which episode this transition belongs to (to get walls and past positions)
            walls = None
            episode_transitions = None
            transition_idx_in_episode = None
            
            for episode in self.episodes:
                if transition in episode['transitions']:
                    walls = episode['walls']
                    episode_transitions = episode['transitions']
                    transition_idx_in_episode = episode_transitions.index(transition)
                    break
            
            if walls is not None:
                # Gather past Pacman positions from this episode
                past_positions = []
                for i in range(max(0, transition_idx_in_episode - memory_length), transition_idx_in_episode):
                    past_positions.append(episode_transitions[i]['state']['pacman_pos'])
                
                state_tensor = self.state_to_tensor(
                    transition['state'], 
                    walls, 
                    past_positions=past_positions,
                    memory_length=memory_length
                )
                states.append(state_tensor)
                actions.append(transition['action'])
                rewards.append(transition['reward'])
        
        states = torch.stack(states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        
        return states, actions, rewards
    
    def get_episode_data(self, episode_idx):
        """Get all data for a specific episode."""
        if episode_idx >= len(self.episodes):
            raise IndexError(f"Episode {episode_idx} does not exist")
        
        return self.episodes[episode_idx]


def main():
    """Demo: Load and display statistics about recorded gameplay."""
    dataset = GameplayDataset()
    dataset.print_statistics()
    
    if TORCH_AVAILABLE and len(dataset.episodes) > 0:
        print("Example: Loading a batch of 8 transitions with memory_length=5...")
        states, actions, rewards = dataset.get_training_batch(batch_size=8, memory_length=5)
        print(f"  States shape: {states.shape}")
        print(f"  Expected shape: (8, 10, height, width) = (batch, 5 base + 5 memory, H, W)")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards.numpy()}")
    elif len(dataset.episodes) > 0:
        print("\nExample transition:")
        t = dataset.transitions[0]
        print(f"  Step: {t['step']}")
        print(f"  Action: {t['action']}")
        print(f"  Reward: {t['reward']}")
        print(f"  Pacman position: {t['state']['pacman_pos']}")
        print(f"  Food remaining: {np.sum(t['state']['food_grid'])}")


if __name__ == '__main__':
    main()
