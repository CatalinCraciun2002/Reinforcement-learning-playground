"""
Epoch Visualizer - Captures all training data for debugging and visualization
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import torch


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
                            'advantage': float (filled after GAE calculation),
                            'actor_loss': float (filled after loss calculation),
                            'critic_loss': float (filled after loss calculation),
                            'entropy': float (filled after loss calculation),
                            'total_loss': float (filled after loss calculation)
                        }
                    ]
                }
            ]
        }
    """
    
    def __init__(self, save_dir: str, hyperparams: Dict[str, Any]):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Base directory to save visualization data
            hyperparams: Training hyperparameters for metadata
        """
        # Save directly to the provided directory (no timestamp subdirectory)
        self.save_path = save_dir
        os.makedirs(self.save_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            'environments': [{'steps': []} for _ in range(batch_size)]
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
            'advantage': None,  # Will be filled later
            'actor_loss': None,  # Will be filled after loss calculation
            'critic_loss': None,  # Will be filled after loss calculation
            'entropy': None,  # Will be filled after loss calculation
            'total_loss': None  # Will be filled after loss calculation
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
    
    def to_float(self, x):
        if torch.is_tensor(x):
            return x.item()
        return float(x)

    def record_losses(self, env_idx: int, step_idx: int, 
                      actor_loss: float, critic_loss: float, 
                      entropy: float, total_loss: float):
        """
        Record loss values for a specific environment and step.
        
        Args:
            env_idx: Index of the environment
            step_idx: Index of the step within that environment
            actor_loss: Actor loss value (scalar or tensor)
            critic_loss: Critic loss value (scalar or tensor)
            entropy: Entropy value (scalar or tensor)
            total_loss: Total loss value (scalar or tensor)
        """
        if self.current_epoch is None:
            raise RuntimeError("Must call start_epoch before recording losses")
        
      
        
        steps = self.epochs[self.current_epoch]['environments'][env_idx]['steps']
        
        if step_idx >= len(steps):
            raise ValueError(f"Step index {step_idx} out of range (max: {len(steps)-1})")
        
        steps[step_idx]['actor_loss'] = self.to_float(actor_loss)
        steps[step_idx]['critic_loss'] = self.to_float(critic_loss)
        steps[step_idx]['entropy'] = self.to_float(entropy)
        steps[step_idx]['total_loss'] = self.to_float(total_loss)
    
    def record_batch_losses(self, actor_loss, critic_loss, entropy, total_loss, 
                           batch_size: int, steps_per_epoch: int):
        """
        Record loss values for all environments and steps in a batch.
        Handles reshaping and iteration internally.
        
        Args:
            actor_loss: Tensor of shape (batch_size * steps_per_epoch,)
            critic_loss: Tensor of shape (batch_size * steps_per_epoch,)
            entropy: Tensor of shape (batch_size * steps_per_epoch,)
            total_loss: Tensor of shape (batch_size * steps_per_epoch,)
            batch_size: Number of parallel environments
            steps_per_epoch: Number of steps per environment
        """
        if self.current_epoch is None:
            raise RuntimeError("Must call start_epoch before recording losses")
        
        import torch
        
        # Reshape losses from (batch_size * steps_per_epoch,) to (batch_size, steps_per_epoch)
        actor_loss_reshaped = actor_loss.view(batch_size, steps_per_epoch)
        critic_loss_reshaped = critic_loss.view(batch_size, steps_per_epoch)
        entropy_reshaped = entropy.view(batch_size, steps_per_epoch)
        total_loss_reshaped = total_loss.view(batch_size, steps_per_epoch)
        
        for env_idx, actor_loss_env, critic_loss_env, entropy_env, total_loss_env in \
            zip(range(batch_size), actor_loss_reshaped, critic_loss_reshaped, entropy_reshaped, total_loss_reshaped):
            
            for step_idx, actor_loss_step, critic_loss_step, entropy_step, total_loss_step in \
                zip(range(steps_per_epoch), actor_loss_env, critic_loss_env, entropy_env, total_loss_env):
                
                self.record_losses(
                    env_idx=env_idx,
                    step_idx=step_idx,
                    actor_loss=actor_loss_step,
                    critic_loss=critic_loss_step,
                    entropy=entropy_step,
                    total_loss=total_loss_step
                )
    
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
