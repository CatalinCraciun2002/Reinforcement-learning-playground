"""
Actor-Critic Training Script with GAE (Generalized Advantage Estimation)
"""

import torch
import torch.optim as optim
import numpy as np
import sys
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_residual_conv import ActorCriticNetwork
from agents.rlAgent import RLAgent
from core.environment import PacmanEnv
from core.game import Directions
from display import graphicsDisplay


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
          pretrained_model_path=None, save_model_path=None):
    """
    Training with GAE.
    
    Args:
        lam: GAE lambda for advantage propagation (default: 0.95)
        pretrained_model_path: Path to pretrained model checkpoint (e.g., from human_feedback training)
        save_model_path: Path to save the trained model (default: inside timestamped run folder)
    """
    
    print("="*60)
    print(f"Training: {num_epochs} epochs, {batch_size} parallel games")
    print(f"Layout: {layout_name}, γ={gamma}, λ={lam}, lr={lr}")
    if pretrained_model_path:
        print(f"Loading pretrained model: {pretrained_model_path}")
    print("="*60)
    
    net = ActorCriticNetwork(memory_context=5)
    
    # Load pretrained weights if provided
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        try:
            net.load_state_dict(torch.load(pretrained_model_path), strict=False)
            print(f"✓ Successfully loaded pretrained model from {pretrained_model_path}")
        except Exception as e:
            print(f"⚠ Warning: Failed to load pretrained model: {e}")
            print("  Continuing with randomly initialized weights...")
    elif pretrained_model_path:
        print(f"⚠ Warning: Pretrained model path specified but file not found: {pretrained_model_path}")
        print("  Continuing with randomly initialized weights...")
    
    agent = RLAgent(net, memory_context=5)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Setup TensorBoard
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/rl_training/{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    print("="*60 + "\n")
    
    # Set default save path inside the run folder if not specified
    if save_model_path is None:
        save_model_path = f'{log_dir}/model_checkpoint.pth'
    
    envs = [PacmanEnv(agent, layout_name) for _ in range(batch_size)]
    
    losses, wins, total_steps = [], 0, 0
    
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in pbar:

        agent.model.train()        
        
        # Store information for loss
        all_log_probs = []
        all_advantages = []
        all_entropies = []
        all_td_errors = []

        for env in envs:

            prev_states = {'td_error': [], 'game_over': []}
            episode_steps = 0

            state = env.game.state
            legal = env.get_legal(state)
            probs, value = agent.forward(state)

            for _ in range(steps_per_epoch):

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
                    # Reset environment for next iteration
                    total_steps += episode_steps
                    episode_steps = 0
                    env.reset()
                    next_state = env.game.state
                else:
                    td_target = reward + gamma * next_value.detach()
                    td_error = td_target - value
                

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
            
            all_td_errors.extend(prev_states['td_error'])
            all_advantages.extend(advantages)
            

        all_log_probs = torch.stack(all_log_probs)
        all_advantages = torch.stack(all_advantages)
        all_entropies = torch.stack(all_entropies)
        all_td_errors = torch.stack(all_td_errors)

        # Normalize advantages for stable training
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Scale critic loss to match actor loss magnitude
        critic_loss = 0.5 * (all_td_errors ** 2).mean()
        actor_loss = -(all_log_probs * all_advantages.detach()).mean()
        entropy_bonus = 0.01 * all_entropies.mean()
        
        total_loss = actor_loss + critic_loss - entropy_bonus

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
        writer.add_scalar('Loss/total', total_loss.item(), epoch)
        writer.add_scalar('Loss/actor', actor_loss.item(), epoch)
        writer.add_scalar('Loss/critic', critic_loss.item(), epoch)
        writer.add_scalar('Loss/entropy_bonus', entropy_bonus.item(), epoch)
        writer.add_scalar('Performance/win_rate', win_rate, epoch)
        writer.add_scalar('Performance/total_wins', wins, epoch)
        writer.add_scalar('Performance/avg_steps', avg_steps, epoch)
        writer.add_scalar('Training/advantage_mean', all_advantages.mean().item(), epoch)
        writer.add_scalar('Training/advantage_std', all_advantages.std().item(), epoch)
        
        pbar.set_postfix({
            'Loss': f'{losses[-1]:.3f}',
            'Wins': wins,
            'AvgSteps': f'{avg_steps:.1f}',
            'WinRate': f'{win_rate:.1f}%'
        })
        
        # Validation
        if (epoch + 1) % show_epochs == 0:
            print("\n" + "="*60)
            print(f"Running validation game at epoch {epoch+1}...")
            print("="*60)
            
            agent.model.eval()
            score, won, steps = run_validation_game(agent, layout_name, with_graphics=True)
            
            # Log validation results
            writer.add_scalar('Validation/score', score, epoch)
            writer.add_scalar('Validation/won', 1 if won else 0, epoch)
            writer.add_scalar('Validation/steps', steps, epoch)
            
            print(f"\nValidation Result - Score: {score}, {'WON!' if won else 'Lost'}, Steps: {steps}")
            print("="*60 + "\n")
            
            # Return to training mode
            agent.model.train()
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Wins: {wins}/{num_epochs*batch_size} ({100*wins/(num_epochs*batch_size):.1f}%)")
    print(f"TensorBoard logs saved to: {log_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(net.state_dict(), save_model_path)
    print(f"Model saved to: {save_model_path}")
    
    return net


if __name__ == "__main__":

    VALIDATE = True
    
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
        pretrained_path = 'runs/human_feedback/20260201_212205/model_checkpoint.pth'
        
        train(num_epochs=100, batch_size=32, steps_per_epoch=20, 
              layout_name='mediumClassic', gamma=0.95, lam=0.95, lr=1e-5, show_epochs=20,
              pretrained_model_path=pretrained_path)
