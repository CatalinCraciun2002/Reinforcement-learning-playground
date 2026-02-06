"""
Train ActorCriticNetwork using Human Gameplay Recordings

This script implements supervised learning (behavioral cloning) to train
the ActorCriticNetwork model to imitate human gameplay. All recorded games
are used for training, and validation is performed by having the model
play games to measure actual performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models.simple_residual_conv import ActorCriticNetwork
from human_feedback.data_loader import GameplayDataset
from runs.logger import TensorBoardLogger
from agents.rlAgent import RLAgent
from core.environment import PacmanEnv


# Action mapping: string to index
ACTION_TO_IDX = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4  # Stop is a separate action
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


def prepare_training_data(dataset, memory_length=5, gamma=0.99):
    """
    Prepare training data from all episodes with returns computed.
    
    Args:
        dataset: GameplayDataset instance
        memory_length: Number of past positions to include
        gamma: Discount factor for computing returns
    
    Returns:
        (train_data, episode_scores) where:
        - train_data: List of dicts with state_tensor, action_idx, return_value
        - episode_scores: List of final scores from each episode
    """
    all_data = []
    episode_scores = []
    
    print("Preparing training data from all games...")
    for episode in tqdm(dataset.episodes, desc="Processing episodes"):
        walls = episode['walls']
        transitions = episode['transitions']
        
        # Get final score from last transition
        if transitions:
            final_score = transitions[-1]['state'].get('score', 0)
            episode_scores.append(final_score)
        
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
    
    print(f"Training samples: {len(all_data)} from {len(episode_scores)} episodes")
    print(f"Average human score: {np.mean(episode_scores):.1f}")
    
    return all_data, episode_scores


def train_epoch(model, train_data, optimizer, device, batch_size=32, train_critic=True):
    """
    Train for one epoch.
    
    Args:
        model: ActorCriticNetwork
        train_data: List of training samples
        optimizer: PyTorch optimizer
        device: Device to train on
        batch_size: Batch size
        train_critic: Whether to train the critic head
    
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
        
        # Critic loss (MSE with normalized returns)
        if train_critic:
            return_mean = returns.mean().detach()
            return_std = returns.std().detach() + 1e-8
            values_norm = (values.squeeze() - return_mean) / return_std
            returns_norm = (returns - return_mean) / return_std
            critic_loss = nn.MSELoss()(values_norm, returns_norm)
        else:
            critic_loss = torch.tensor(0.0)
        
        # Total loss with coefficients
        total_loss = 1.0 * actor_loss + 0.5 * critic_loss
        
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




def validate(model, layout_name='mediumClassic', memory_context=5, num_games=32, max_steps=1000):
    """
    Validate the model by playing num_games without graphics.
    
    Args:
        model: ActorCriticNetwork
        layout_name: Name of the layout to play
        memory_context: Memory context for the agent
        num_games: Number of games to play for validation
        max_steps: Maximum steps per game
    
    Returns:
        avg_score: Average score across all validation games
    """
    model.eval()
    scores = []
    
    for _ in range(num_games):
        agent = RLAgent(model, memory_context=memory_context)
        val_env = PacmanEnv(agent, layout_name, display=None)
        val_env.reset()
        
        steps = 0
        game_done = False
        
        while not game_done and steps < max_steps:
            state = val_env.game.state
            legal = val_env.get_legal(state)
            
            if not legal:
                break
            
            with torch.no_grad():
                probs, _ = agent.forward(state)
            
            action, action_idx = agent.getAction(legal, probs)
            _, reward, game_done = val_env.step(action)
            steps += 1
        
        score = val_env.game.state.getScore()
        scores.append(score)
    
    avg_score = np.mean(scores)
    return avg_score


def train(data_dir='game_runs_data', num_epochs=50, batch_size=32, lr=1e-4,
          memory_length=5, train_critic=True, gamma=0.99, layout_name='mediumClassic',
          validation_games=8, resume_from_checkpoint=None):
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
        layout_name: Layout to use for validation games
        validation_games: Number of validation games to play each epoch (default: 8)
        resume_from_checkpoint: Path to checkpoint to resume training from (optional)
    """
    # Set random seeds for reproducible shuffling
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("="*60)
    print("Training ActorCriticNetwork from Human Gameplay")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Memory length: {memory_length}, Train critic: {train_critic}")
    print(f"Validation: {validation_games} games on {layout_name}")
    print("="*60 + "\n")
    
    # Load dataset
    dataset = GameplayDataset(data_dir)
    
    if len(dataset.episodes) == 0:
        print("Error: No gameplay recordings found!")
        return
    
    dataset.print_statistics()
    
    # Prepare training data (all games)
    train_data, episode_scores = prepare_training_data(dataset, memory_length, gamma=gamma)
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model = ActorCriticNetwork(memory_context=memory_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup logger with hyperparameters
    hyperparams = {
        'batch_size': batch_size,
        'learning_rate': lr,
        'memory_length': memory_length,
        'gamma': gamma,
        'train_critic': train_critic,
        'num_epochs': num_epochs,
        'layout': layout_name
    }
    
    logger = TensorBoardLogger(
        training_type='human_feedback',
        pretrained_model_path=resume_from_checkpoint,
        hyperparams=hyperparams
    )
    
    logger.print_header()
    
    # Load checkpoint (if provided)
    start_epoch, best_val_score = logger.load_checkpoint(model, optimizer)
    
    # Setup TensorBoard
    writer, log_dir, is_resuming = logger.setup_tensorboard()
    
    # Get checkpoint paths
    best_checkpoint_path, last_checkpoint_path = logger.get_checkpoint_paths()
    
    # Training loop
    end_epoch = start_epoch + num_epochs
    
    pbar = tqdm(range(start_epoch, end_epoch), desc="Training", unit="epoch", initial=start_epoch, total=end_epoch)
    
    for epoch in pbar:
        # Train
        train_actor_loss, train_critic_loss, train_acc = train_epoch(
            model, train_data, optimizer, device, batch_size, train_critic
        )
        
        # Validate by playing games
        avg_val_score = validate(model, layout_name, memory_length, validation_games)
        
        # Log to TensorBoard
        logger.log_scalars({
            'Loss/train_actor': train_actor_loss,
            'Loss/train_critic': train_critic_loss,
            'Accuracy/train': train_acc,
            'Score/validation': avg_val_score
        }, epoch)
        
        # Update progress bar
        pbar.set_postfix({
            'TrainLoss': f'{train_actor_loss:.3f}',
            'TrainAcc': f'{train_acc:.3f}',
            'ValScore': f'{avg_val_score:.0f}'
        })
        
        # Save checkpoints
        is_best = avg_val_score > best_val_score
        if is_best:
            best_val_score = avg_val_score
            pbar.write(f"✓ Epoch {epoch+1}: New best validation score: {avg_val_score:.0f}")
            
        logger.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metric_value=avg_val_score,
            metric_name='val_score',
            is_best=is_best
        )
    
    # Close logger and print summary
    logger.close()
    
    logger.print_completion_summary({
        'Best Validation Score': f"{best_val_score:.0f}"
    })
    print()
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ActorCriticNetwork from human gameplay')
    parser.add_argument('--data-dir', type=str, default='game_runs_data',
                       help='Directory containing gameplay recordings')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training and number of validation games')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--memory-length', type=int, default=5,
                       help='Number of past positions to include')
    parser.add_argument('--no-critic', action='store_true',
                       help='Do not train the critic head')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor for computing returns')
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Layout to use for validation games')
    parser.add_argument('--validation-games', type=int, default=8,
                       help='Number of validation games to play each epoch')
    parser.add_argument('--resume', type=str, default='runs\\human_feedback\\20260203_191824',
                       help='Path to checkpoint or run directory to resume from (uses model_last.pth)')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        memory_length=args.memory_length,
        train_critic=not args.no_critic,
        gamma=args.gamma,
        layout_name=args.layout,
        validation_games=args.validation_games,
        resume_from_checkpoint=args.resume
    )
