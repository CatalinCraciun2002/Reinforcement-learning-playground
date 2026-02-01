"""
Train ActorCriticNetwork using Human Gameplay Recordings

This script implements supervised learning (behavioral cloning) to train
the ActorCriticNetwork model to imitate human gameplay.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.simple_residual_conv import ActorCriticNetwork
from human_feedback.data_loader import GameplayDataset


# Action mapping: string to index
ACTION_TO_IDX = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 0  # Map Stop to North (or could be handled separately)
}


def compute_discounted_returns(episode_transitions, gamma=0.99):
    """
    Compute discounted returns for an episode.
    
    Args:
        episode_transitions: List of transitions from a single episode
        gamma: Discount factor
    
    Returns:
        List of discounted returns for each transition
    """
    returns = []
    G = 0
    
    for transition in reversed(episode_transitions):
        G = transition['reward'] + gamma * G
        returns.insert(0, G)
    
    return returns


def prepare_training_data(dataset, memory_length=5, train_split=0.9, gamma=0.99):
    """
    Prepare training and validation data with returns computed.
    
    Args:
        dataset: GameplayDataset instance
        memory_length: Number of past positions to include
        train_split: Fraction of data to use for training
        gamma: Discount factor for computing returns
    
    Returns:
        (train_data, val_data) where each is a list of dicts with:
        - state_tensor: Tensor with memory context
        - action_idx: Integer action index
        - return_value: Discounted return
    """
    all_data = []
    
    print("Preparing training data with returns...")
    for episode in tqdm(dataset.episodes, desc="Processing episodes"):
        walls = episode['walls']
        transitions = episode['transitions']
        
        # Compute returns for this episode
        returns = compute_discounted_returns(transitions, gamma)
        
        # Process each transition
        for i, (transition, G) in enumerate(zip(transitions, returns)):
            # Gather past positions
            past_positions = []
            for j in range(max(0, i - memory_length), i):
                past_positions.append(transitions[j]['state']['pacman_pos'])
            
            # Convert state to tensor
            state_tensor = dataset.state_to_tensor(
                transition['state'],
                walls,
                past_positions=past_positions,
                memory_length=memory_length
            )
            
            # Convert action to index
            action_str = transition['action']
            action_idx = ACTION_TO_IDX.get(action_str, 0)
            
            all_data.append({
                'state': state_tensor,
                'action': action_idx,
                'return': G
            })
    
    # Shuffle and split
    np.random.shuffle(all_data)
    split_idx = int(len(all_data) * train_split)
    
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    return train_data, val_data


def train_epoch(model, train_data, optimizer, device, batch_size=32, train_critic=True, writer=None, epoch=0):
    """
    Train for one epoch.
    
    Args:
        model: ActorCriticNetwork
        train_data: List of training samples
        optimizer: PyTorch optimizer
        device: Device to train on
        batch_size: Batch size
        train_critic: Whether to train the critic head
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch number for logging
    
    Returns:
        (avg_actor_loss, avg_critic_loss, accuracy)
    """
    model.train()
    
    # Shuffle data
    np.random.shuffle(train_data)
    
    total_actor_loss = 0
    total_critic_loss = 0
    total_correct = 0
    num_batches = 0
    
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        
        # Prepare batch tensors
        states = torch.stack([item['state'] for item in batch]).to(device)
        actions = torch.tensor([item['action'] for item in batch], dtype=torch.long).to(device)
        returns = torch.tensor([item['return'] for item in batch], dtype=torch.float32).to(device)
        
        # Forward pass
        action_probs, values = model(states, return_both=True)
        
        # Actor loss (cross-entropy)
        actor_loss = nn.CrossEntropyLoss()(action_probs, actions)
        
        # Critic loss (MSE with returns)
        critic_loss = nn.MSELoss()(values.squeeze(), returns) if train_critic else torch.tensor(0.0)
        
        # Total loss
        total_loss = actor_loss + (0.5 * critic_loss if train_critic else 0)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item() if train_critic else 0
        
        predicted_actions = torch.argmax(action_probs, dim=1)
        total_correct += (predicted_actions == actions).sum().item()
        
        num_batches += 1
    
    avg_actor_loss = total_actor_loss / num_batches
    avg_critic_loss = total_critic_loss / num_batches if train_critic else 0
    accuracy = total_correct / len(train_data)
    
    return avg_actor_loss, avg_critic_loss, accuracy


def validate(model, val_data, device, batch_size=32, train_critic=True, writer=None, epoch=0):
    """
    Validate the model.
    
    Args:
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch number for logging
    
    Returns:
        (avg_actor_loss, avg_critic_loss, accuracy)
    """
    model.eval()
    
    total_actor_loss = 0
    total_critic_loss = 0
    total_correct = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i + batch_size]
            
            states = torch.stack([item['state'] for item in batch]).to(device)
            actions = torch.tensor([item['action'] for item in batch], dtype=torch.long).to(device)
            returns = torch.tensor([item['return'] for item in batch], dtype=torch.float32).to(device)
            
            action_probs, values = model(states, return_both=True)
            
            actor_loss = nn.CrossEntropyLoss()(action_probs, actions)
            critic_loss = nn.MSELoss()(values.squeeze(), returns) if train_critic else torch.tensor(0.0)
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item() if train_critic else 0
            
            predicted_actions = torch.argmax(action_probs, dim=1)
            total_correct += (predicted_actions == actions).sum().item()
            
            num_batches += 1
    
    avg_actor_loss = total_actor_loss / num_batches
    avg_critic_loss = total_critic_loss / num_batches if train_critic else 0
    accuracy = total_correct / len(val_data)
    
    return avg_actor_loss, avg_critic_loss, accuracy


def train(data_dir='game_runs_data', num_epochs=50, batch_size=32, lr=1e-4,
          memory_length=5, train_critic=True, gamma=0.99, save_path=None):
    """
    Train the ActorCriticNetwork using human gameplay recordings.
    
    Args:
        data_dir: Directory containing gameplay recordings
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        memory_length: Number of past positions to include
        train_critic: Whether to train the critic head
        gamma: Discount factor for computing returns
        save_path: Path to save the trained model (default: inside timestamped run folder)
    """
    print("="*60)
    print("Training ActorCriticNetwork from Human Gameplay")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Memory length: {memory_length}, Train critic: {train_critic}")
    print("="*60 + "\n")
    
    # Load dataset
    dataset = GameplayDataset(data_dir)
    
    if len(dataset.episodes) == 0:
        print("Error: No gameplay recordings found!")
        return
    
    dataset.print_statistics()
    
    # Prepare data
    train_data, val_data = prepare_training_data(dataset, memory_length, gamma=gamma)
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    model = ActorCriticNetwork(memory_context=memory_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup TensorBoard
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/human_feedback/{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}\n")
    
    # Set default save path inside the run folder if not specified
    if save_path is None:
        save_path = f'{log_dir}/model_checkpoint.pth'
    
    # Training loop
    best_val_accuracy = 0
    
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in pbar:
        # Train
        train_actor_loss, train_critic_loss, train_acc = train_epoch(
            model, train_data, optimizer, device, batch_size, train_critic, writer, epoch
        )
        
        # Validate
        val_actor_loss, val_critic_loss, val_acc = validate(
            model, val_data, device, batch_size, train_critic, writer, epoch
        )
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train_actor', train_actor_loss, epoch)
        writer.add_scalar('Loss/train_critic', train_critic_loss, epoch)
        writer.add_scalar('Loss/val_actor', val_actor_loss, epoch)
        writer.add_scalar('Loss/val_critic', val_critic_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Update progress bar
        pbar.set_postfix({
            'TrainAcc': f'{train_acc:.3f}',
            'ValAcc': f'{val_acc:.3f}',
            'TrainLoss': f'{train_actor_loss:.3f}',
            'ValLoss': f'{val_actor_loss:.3f}'
        })
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), save_path)
            pbar.write(f"✓ Epoch {epoch+1}: New best validation accuracy: {val_acc:.3f}")
    
    writer.close()
    
    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print(f"Model saved to: {save_path}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("="*60)
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ActorCriticNetwork from human gameplay')
    parser.add_argument('--data-dir', type=str, default='game_runs_data',
                       help='Directory containing gameplay recordings')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--memory-length', type=int, default=5,
                       help='Number of past positions to include')
    parser.add_argument('--no-critic', action='store_true',
                       help='Do not train the critic head')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor for computing returns')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save the trained model (default: saved inside the timestamped run folder)')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        memory_length=args.memory_length,
        train_critic=not args.no_critic,
        gamma=args.gamma,
        save_path=args.save_path
    )
