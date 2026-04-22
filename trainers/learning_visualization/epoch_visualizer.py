"""
Epoch Visualizer - Captures all training data for debugging and visualization
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class EpochVisualizer:
    """
    Captures and saves all training data for visualization and debugging.
    
    Data structure:
        epochs[epoch_num] = {
            'environments': [
                {  # For each environment in batch
                    'steps': [
                        {  # For each step
                            'state': serialized game state,
                            'legal_actions': list of legal action strings,
                            'action_probs': dict mapping action -> probability,
                            'selected_action': str,
                            'selected_action_idx': int,
                            'value': float (critic prediction),
                            'reward': float,
                            'next_value': float,
                            'td_error': float,
                            'td_target': float or None (None if terminal),
                            'done': bool,
                            'advantage': float (filled after GAE calculation)
                        }
                    ]
                }
            ],
            'losses': {
                'actor': float,
                'critic': float,
                'entropy_bonus': float,
                'total': float
            }
        }
    """
    
    def __init__(self, save_dir: str, hyperparams: Dict[str, Any]):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Base directory to save visualization data
            hyperparams: Training hyperparameters for metadata
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(save_dir, f"vis_{timestamp}")
        os.makedirs(self.save_path, exist_ok=True)
        
        self.hyperparams = hyperparams
        self.epochs = {}
        self.current_epoch = None
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'hyperparams': hyperparams
        }
        with open(os.path.join(self.save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Visualization data will be saved to: {self.save_path}")
    
    def start_epoch(self, epoch_num: int, batch_size: int):
        """Start recording a new epoch."""
        self.current_epoch = epoch_num
        self.epochs[epoch_num] = {
            'environments': [{'steps': []} for _ in range(batch_size)],
            'losses': None
        }
    
    def record_step(self, env_idx: int, step_data: Dict[str, Any]):
        """
        Record data for a single step.
        
        Args:
            env_idx: Index of the environment in the batch
            step_data: Dictionary containing:
                - state: game state object
                - legal_actions: list of legal action strings
                - action_probs: tensor or dict of action probabilities
                - selected_action: the chosen action string
                - selected_action_idx: index of the chosen action
                - value: critic's value prediction (tensor or float)
                - reward: reward received
                - next_value: next state value prediction (tensor or float)
                - td_error: TD error (tensor or float)
                - td_target: TD target (tensor or float or None)
                - done: whether episode ended
        """
        if self.current_epoch is None:
            raise RuntimeError("Must call start_epoch before recording steps")
        
        # Serialize game state
        state_data = self._serialize_state(step_data['state'])
        
        # Convert tensors to floats
        import torch
        
        def to_float(x):
            if torch.is_tensor(x):
                return x.item()
            return float(x) if x is not None else None
        
        # Convert action probs to dict if it's a tensor
        action_probs = step_data['action_probs']
        if torch.is_tensor(action_probs):
            # Map to legal actions
            legal_actions = step_data['legal_actions']
            action_probs_dict = {
                action: to_float(action_probs[i]) 
                for i, action in enumerate(legal_actions)
            }
        else:
            action_probs_dict = {k: to_float(v) for k, v in action_probs.items()}
        
        step_record = {
            'state': state_data,
            'legal_actions': step_data['legal_actions'],
            'action_probs': action_probs_dict,
            'selected_action': step_data['selected_action'],
            'selected_action_idx': step_data['selected_action_idx'],
            'value': to_float(step_data['value']),
            'reward': to_float(step_data['reward']),
            'next_value': to_float(step_data['next_value']),
            'td_error': to_float(step_data['td_error']),
            'td_target': to_float(step_data.get('td_target')),
            'done': step_data['done'],
            'advantage': None  # Will be filled later
        }
        
        self.epochs[self.current_epoch]['environments'][env_idx]['steps'].append(step_record)
    
    def record_advantages(self, env_idx: int, advantages: List[float]):
        """
        Record advantages after GAE calculation.
        
        Args:
            env_idx: Index of the environment
            advantages: List of advantage values (one per step)
        """
        if self.current_epoch is None:
            raise RuntimeError("Must call start_epoch before recording advantages")
        
        import torch
        
        steps = self.epochs[self.current_epoch]['environments'][env_idx]['steps']
        
        if len(advantages) != len(steps):
            raise ValueError(f"Mismatch: {len(advantages)} advantages for {len(steps)} steps")
        
        for step, adv in zip(steps, advantages):
            if torch.is_tensor(adv):
                step['advantage'] = adv.item()
            else:
                step['advantage'] = float(adv)
    
    def record_losses(self, actor_loss: float, critic_loss: float, 
                      entropy_bonus: float, total_loss: float):
        """Record loss values for the current epoch."""
        if self.current_epoch is None:
            raise RuntimeError("Must call start_epoch before recording losses")
        
        import torch
        
        def to_float(x):
            if torch.is_tensor(x):
                return x.item()
            return float(x)
        
        self.epochs[self.current_epoch]['losses'] = {
            'actor': to_float(actor_loss),
            'critic': to_float(critic_loss),
            'entropy_bonus': to_float(entropy_bonus),
            'total': to_float(total_loss)
        }
    
    def end_epoch(self):
        """Mark the end of the current epoch and save data."""
        if self.current_epoch is None:
            return
        
        # Save this epoch's data
        epoch_file = os.path.join(self.save_path, f'epoch_{self.current_epoch:04d}.pkl')
        with open(epoch_file, 'wb') as f:
            pickle.dump(self.epochs[self.current_epoch], f)
        
        # Clear from memory to save space (keep only metadata in memory)
        del self.epochs[self.current_epoch]
        self.current_epoch = None
    
    def _serialize_state(self, state) -> Dict[str, Any]:
        """
        Serialize a game state for storage.
        
        Returns a dictionary with all necessary information to reconstruct
        the visual state of the game.
        """
        # Get essential state information
        return {
            'score': state.getScore(),
            'pacman_pos': state.getPacmanPosition(),
            'ghost_positions': state.getGhostPositions(),
            'ghost_states': [(g.getPosition(), g.scaredTimer) for g in state.getGhostStates()],
            'food': state.getFood().data,  # Grid of food
            'capsules': state.getCapsules(),
            'walls': state.getWalls().data,  # Grid of walls
            'is_win': state.isWin(),
            'is_lose': state.isLose(),
            # Store the full state object for replay if needed
            '_full_state': state
        }
    
    def get_save_path(self) -> str:
        """Return the directory where data is being saved."""
        return self.save_path


def load_visualization_data(data_dir: str) -> Dict[str, Any]:
    """
    Load visualization data from a saved directory.
    
    Args:
        data_dir: Path to the visualization data directory
        
    Returns:
        Dictionary with 'metadata' and 'epochs' keys
    """
    # Load metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Find all epoch files
    epoch_files = sorted([
        f for f in os.listdir(data_dir) 
        if f.startswith('epoch_') and f.endswith('.pkl')
    ])
    
    epochs = {}
    for epoch_file in epoch_files:
        # Extract epoch number
        epoch_num = int(epoch_file.replace('epoch_', '').replace('.pkl', ''))
        
        # Load epoch data
        with open(os.path.join(data_dir, epoch_file), 'rb') as f:
            epochs[epoch_num] = pickle.load(f)
    
    return {
        'metadata': metadata,
        'epochs': epochs
    }
