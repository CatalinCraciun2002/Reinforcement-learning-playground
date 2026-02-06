"""
Actor-Critic Training Script with GAE (Generalized Advantage Estimation)
"""

import torch
import torch.optim as optim
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_residual_conv import ActorCriticNetwork
from agents.rlAgent import RLAgent
from core.environment import PacmanEnv
from core.game import Directions
from display import graphicsDisplay
from runs.logger import TensorBoardLogger
from reinforcement_learning.epoch_visualizer import EpochVisualizer


def run_validation_game(agent, layout_name='mediumClassic', with_graphics=True, max_steps=1000):
    """Run a validation game and return score, win status, and steps taken."""
    display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.05) if with_graphics else None
    val_env = PacmanEnv(agent, layout_name, display)
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
    won = val_env.game.state.isWin()
    
    return score, won, steps


def train(num_epochs=100, batch_size=32, steps_per_epoch=20, 
          layout_name='mediumClassic', gamma=0.95, lam=0.95, lr=1e-5, show_epochs=50,
          validation_games=8, pretrained_model_path=None, use_best_checkpoint=False,
          save_visualization_data=False):
    """
    Training with GAE.
    
    Args:
        validation_games: Number of validation games to play each epoch (default: 8)
        lam: GAE lambda for advantage propagation (default: 0.95)
        pretrained_model_path: Path to checkpoint to load (can be from human_feedback or previous RL training)
        use_best_checkpoint: If True and pretrained_model_path is a directory, load model_best.pth; 
                            otherwise load model_last.pth (default: False)
        save_visualization_data: If True, save all training data for visualization (default: False)
    """
    
    # Setup logger with hyperparameters
    hyperparams = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'steps_per_epoch': steps_per_epoch,
        'learning_rate': lr,
        'gamma': gamma,
        'lambda': lam,
        'layout': layout_name,
        'pretrained': pretrained_model_path is not None,
        'use_best_checkpoint': use_best_checkpoint
    }
    
    logger = TensorBoardLogger(
        training_type='rl_training',
        pretrained_model_path=pretrained_model_path,
        hyperparams=hyperparams
    )
    
    logger.print_header()
    
    # Create model and optimizer
    net = ActorCriticNetwork(memory_context=5)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Load checkpoint (if provided)
    start_epoch, best_win_rate = logger.load_checkpoint(net, optimizer, use_best_checkpoint=use_best_checkpoint)
    
    agent = RLAgent(net, memory_context=5)
    
    # Setup TensorBoard
    writer, log_dir, is_resuming = logger.setup_tensorboard()
    
    # Get checkpoint paths
    best_checkpoint_path, last_checkpoint_path = logger.get_checkpoint_paths()
    
    # Initialize visualizer if enabled
    visualizer = None
    if save_visualization_data:
        vis_dir = os.path.join(os.path.dirname(__file__), 'visualization_data')
        os.makedirs(vis_dir, exist_ok=True)
        visualizer = EpochVisualizer(vis_dir, hyperparams)
    
    envs = [PacmanEnv(agent, layout_name) for _ in range(batch_size)]
    
    losses, wins, total_steps = [], 0, 0
    
    # Evaluate at epoch 0 if starting from a checkpoint (for baseline)
    if start_epoch > 0 or pretrained_model_path:
        print("=" * 60)
        print("Computing epoch 0 baseline metrics...")
        print("=" * 60)
        
        agent.model.eval()
        
        # Run validation games to get initial metrics
        baseline_scores = []
        baseline_wins = []
        for _ in range(validation_games):
            score, won, steps = run_validation_game(agent, layout_name, with_graphics=False)
            baseline_scores.append(score)
            baseline_wins.append(1 if won else 0)
        
        avg_score = sum(baseline_scores) / len(baseline_scores)
        win_rate = sum(baseline_wins) / len(baseline_wins)
        
        # Log baseline metrics
        logger.log_scalars({
            'Score/score': avg_score,
            'Score/won': win_rate,
        }, 0)
        
        print(f"Baseline - Avg Score: {avg_score:.1f}, Win Rate: {win_rate:.1%}")
        print("=" * 60 + "\n")
        
        agent.model.train()
    elif start_epoch == 0:
        print("Starting training from scratch (no baseline metrics)")
        print("=" * 60 + "\n")
    
    # When resuming, train for num_epochs additional epochs beyond start_epoch
    end_epoch = start_epoch + num_epochs
    
    pbar = tqdm(range(start_epoch, end_epoch), desc="Training", unit="epoch", initial=start_epoch, total=end_epoch)
    
    for epoch in pbar:

        agent.model.train()        
        
        # Initialize visualizer for this epoch
        if visualizer:
            visualizer.start_epoch(epoch, batch_size)
        
        # Store information for loss
        all_log_probs = []
        all_advantages = []
        all_entropies = []
        all_td_errors = []

        for env_idx, env in enumerate(envs):

            prev_states = {'td_error': [], 'game_over': []}
            episode_steps = 0

            state = env.game.state
            legal = env.get_legal(state)
            probs, value = agent.forward(state)

            for step_idx in range(steps_per_epoch):

                action, action_idx = agent.getAction(legal, probs)

                prob_entropy = -(probs * torch.log(probs + 1e-10)).sum()
                all_entropies.append(prob_entropy)

                log_prob = torch.log(probs[action_idx] + 1e-10)
                all_log_probs.append(log_prob)            

                # Execute next state
                next_state, reward, done = env.step(action)
                next_probs, next_value = agent.forward(next_state)
                
                episode_steps += 1

                if done:
                    td_error = reward - value
                    td_target = None
                    # Reset environment for next iteration
                    total_steps += episode_steps
                    episode_steps = 0
                    env.reset()
                    next_state = env.game.state
                else:
                    td_target = reward + gamma * next_value.detach()
                    td_error = td_target - value
                
                # Record visualization data
                if visualizer:
                    step_data = {
                        'state': state,
                        'legal_actions': legal,
                        'action_probs': probs,
                        'selected_action': action,
                        'selected_action_idx': action_idx,
                        'value': value,
                        'reward': reward,
                        'next_value': next_value,
                        'td_error': td_error,
                        'td_target': td_target,
                        'done': done
                    }
                    visualizer.record_step(env_idx, step_data)

                prev_states['td_error'].append(td_error)
                prev_states['game_over'].append(done)

                state = next_state
                legal = env.get_legal(state)
                probs, value = next_probs, next_value

            # Calculate GAE
            advantages = []
            gae = 0
            
            for done, td_error in zip(reversed(prev_states['game_over']), reversed(prev_states['td_error'])):
                
                if done:
                    advantage = td_error
                else:
                    advantage = td_error + gamma * lam * gae
                
                advantages.append(advantage)
                gae = advantage.detach()


            # Reverse advantages to match original order
            advantages.reverse()
            
            # Record advantages for visualization
            if visualizer:
                visualizer.record_advantages(env_idx, advantages)
            
            all_td_errors.extend(prev_states['td_error'])
            all_advantages.extend(advantages)
            

        all_log_probs = torch.stack(all_log_probs)
        all_advantages = torch.stack(all_advantages)
        all_entropies = torch.stack(all_entropies)
        all_td_errors = torch.stack(all_td_errors)

        # Normalize advantages for stable training
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Normalize TD errors for critic loss (brings loss to comparable scale)
        td_mean = all_td_errors.mean().detach()
        td_std = all_td_errors.std().detach() + 1e-8
        all_td_errors_norm = (all_td_errors - td_mean) / td_std
        
        # Critic loss: MSE on normalized TD errors (range ~[0, 4])
        critic_loss = (all_td_errors_norm ** 2).mean()
        
        # Actor loss
        actor_loss = -(all_log_probs * all_advantages.detach()).mean()
        
        # Entropy bonus
        entropy_bonus = all_entropies.mean()
        
        # Total loss with coefficients
        total_loss = 1.0 * actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
        
        # Record losses for visualization
        if visualizer:
            visualizer.record_losses(actor_loss, critic_loss, entropy_bonus, total_loss)

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        wins += sum([env.wins for env in envs])
        losses.append(total_loss.item())
        
        avg_steps = total_steps / sum([env.wins + 1 for env in envs]) if total_steps > 0 else 0
        win_rate = 100*wins/((epoch+1)*batch_size)
        
        # Log to TensorBoard
        logger.log_scalars({
            'Loss/total': total_loss.item(),
            'Loss/actor': actor_loss.item(),
            'Loss/critic': critic_loss.item(),
            'Loss/entropy_bonus': entropy_bonus.item(),
            'Performance/total_wins': wins,
            'Performance/avg_steps': avg_steps,
        }, epoch)
        

        
        # Validation every epoch (without graphics), with graphics only at intervals
        agent.model.eval()
        
        # Run multiple validation games and average
        val_scores = []
        val_wins = []
        for _ in range(validation_games):
            score, won, steps = run_validation_game(agent, layout_name, with_graphics=False)
            val_scores.append(score)
            val_wins.append(1 if won else 0)
        
        val_score = sum(val_scores) / len(val_scores)
        val_won = sum(val_wins) / len(val_wins)
        
        agent.model.train()
        
        # Log validation results every epoch
        logger.log_scalars({
            'Score/score': val_score,
            'Score/won': val_won
        }, epoch)

        pbar.set_postfix({
            'Actor': f'{actor_loss.item():.4f}',
            'Critic': f'{critic_loss.item():.2f}',
            'ValScore': f'{val_score:.1f}'
        })
        
        # Display validation with graphics at specified intervals
        if epoch % show_epochs == 0:
            print("\n" + "="*60)
            print(f"Running validation game with graphics at epoch {epoch+1}...")
            print("="*60)
            
            agent.model.eval()
            score, won, steps = run_validation_game(agent, layout_name, with_graphics=True)
            
            print(f"\nValidation Result - Score: {score}, {'WON!' if won else 'Lost'}, Steps: {steps}")
            print("="*60 + "\n")
            
            agent.model.train()

        # Save checkpoints based on validation score
        val_win_rate = val_won  # Already averaged from multiple games
        is_best = val_win_rate > best_win_rate
        if is_best:
            best_win_rate = val_win_rate
            
        logger.save_checkpoint(
            epoch=epoch,
            model=net,
            optimizer=optimizer,
            metric_value=val_win_rate,
            metric_name='win_rate',
            is_best=is_best,
            additional_data={'wins': wins, 'total_steps': total_steps, 'val_win_rate': val_win_rate}
        )
        
        # Save visualization data for this epoch
        if visualizer:
            visualizer.end_epoch()
    
    # Close logger and print summary
    logger.close()
    
    final_win_rate = f"{wins}/{(end_epoch)*batch_size} = {wins/((end_epoch)*batch_size) if (end_epoch)*batch_size > 0 else 0:.2%}"
    logger.print_completion_summary({
        'Final Win Rate': final_win_rate,
        'Best Win Rate': f"{best_win_rate:.2%}",
        'Total Steps': total_steps
    })
    
    return net


if __name__ == "__main__":

    VALIDATE = False
    
    if VALIDATE:
        
        net = ActorCriticNetwork(memory_context=5)
        path = 'runs/human_feedback/20260201_212205/model_checkpoint.pth'
        net.load_state_dict(torch.load(path), strict=False)
        net.eval()
        
        agent = RLAgent(net, memory_context=5)
        
        score, won, steps = run_validation_game(agent, 'mediumClassic', with_graphics=True)
        print(f"Score: {score}, Won: {won}, Steps: {steps}")
    else:
        # Load pretrained model from human feedback (set to None to train from scratch)
        pretrained_path = 'runs\human_feedback\\20260203_181930'
        
        train(num_epochs=20, batch_size=32, steps_per_epoch=20, 
              layout_name='mediumClassic', gamma=0.95, lam=0.80, lr=1e-5, show_epochs=20,
              pretrained_model_path=pretrained_path, use_best_checkpoint=True, save_visualization_data=True)
