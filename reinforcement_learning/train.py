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

from reinforcement_learning.model import ActorCriticNetwork
from reinforcement_learning.agent import RLAgent
from reinforcement_learning.environment import PacmanEnv
from core.game import Directions
from display import graphicsDisplay


def train(num_epochs=100, batch_size=32, steps_per_epoch=20, 
          layout_name='mediumClassic', gamma=0.95, lam=0.95, lr=1e-5, show_epochs=50):
    """
    Training with GAE.
    
    Args:
        lam: GAE lambda for advantage propagation (default: 0.95)
    """
    
    print("="*60)
    print(f"Training: {num_epochs} epochs, {batch_size} parallel games")
    print(f"Layout: {layout_name}, γ={gamma}, λ={lam}, lr={lr}")
    print("="*60)
    
    net = ActorCriticNetwork(memory_context=5)
    agent = RLAgent(memory_context=5)
    agent.model = net
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    envs = [PacmanEnv(agent, layout_name) for _ in range(batch_size)]
    
    losses, wins = [], 0
    
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        net.train()
        optimizer.zero_grad()
        
        epoch_loss, epoch_wins = 0, 0
        
        # Store transitions for GAE
        transitions = []
        
        for _ in range(steps_per_epoch):
            for env_idx, env in enumerate(envs):
                if env.is_over:
                    if env.game.state.isWin():
                        epoch_wins += 1
                    state = env.reset()
                else:
                    state = env.game.state
                
                legal = state.getLegalPacmanActions()
                if Directions.STOP in legal:
                    legal.remove(Directions.STOP)
                if not legal:
                    continue
                
                # Update position buffer
                agent.update_position_buffer(state)
                
                # Forward pass
                state_tensor = agent.state_to_tensor(state)
                probs, value = net(state_tensor, return_both=True)
                probs = probs.squeeze()
                value = value.squeeze()
                
                # Sample action
                mask = torch.tensor([1.0 if a in legal else 0.0 for a in agent.actions])
                masked = probs * mask
                masked = masked / masked.sum() if masked.sum() > 0 else mask / mask.sum()
                action_idx = torch.multinomial(masked, 1).item()
                action = agent.actions[action_idx]
                
                # Execute
                next_state, reward, done = env.step(action)
                
                # Store transition
                transitions.append({
                    'env_idx': env_idx,
                    'value': value,
                    'reward': reward,
                    'done': done,
                    'probs': probs,
                    'action_idx': action_idx
                })
        
        # Compute GAE advantages
        advantages = []
        prev_advantage = {i: 0 for i in range(batch_size)}
        
        for t in reversed(transitions):
            if t['done']:
                td_target = t['reward']
                advantage = td_target - t['value'].detach()
                prev_advantage[t['env_idx']] = 0
            else:
                # Get next value
                next_env = envs[t['env_idx']]
                agent.update_position_buffer(next_env.game.state)
                next_tensor = agent.state_to_tensor(next_env.game.state)
                with torch.no_grad():
                    _, next_value = net(next_tensor, return_both=True)
                
                td_target = t['reward'] + gamma * next_value.squeeze() - 1.0
                td_error = td_target - t['value'].detach()
                
                # GAE: advantage = td_error + λ * γ * prev_advantage
                advantage = td_error + lam * gamma * prev_advantage[t['env_idx']]
                prev_advantage[t['env_idx']] = advantage.item()
            
            advantages.append(advantage)
        
        advantages.reverse()
        
        # Compute losses
        for i, t in enumerate(transitions):
            # Critic loss
            if t['done']:
                target = t['reward']
            else:
                next_env = envs[t['env_idx']]
                agent.update_position_buffer(next_env.game.state)
                next_tensor = agent.state_to_tensor(next_env.game.state)
                with torch.no_grad():
                    _, next_value = net(next_tensor, return_both=True)
                target = t['reward'] + gamma * next_value.squeeze() - 1.0
            
            critic_loss = (t['value'] - target) ** 2
            
            # Actor loss with GAE advantage
            log_prob = torch.log(t['probs'][t['action_idx']] + 1e-10)
            entropy = -(t['probs'] * torch.log(t['probs'] + 1e-10)).sum()
            actor_loss = -(log_prob * advantages[i]) - 0.01 * entropy
            
            loss = actor_loss + critic_loss
            loss.backward()
            epoch_loss += loss.item()
        
        # Update
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        wins += epoch_wins
        losses.append(epoch_loss / len(transitions) if transitions else 0)
        
        pbar.set_postfix({
            'Loss': f'{losses[-1]:.3f}',
            'Wins': epoch_wins,
            'WinRate': f'{100*wins/((epoch+1)*batch_size):.1f}%'
        })
        
        # Validation
        if (epoch + 1) % show_epochs == 0:
            net.eval()
            val_env = PacmanEnv(agent, layout_name, graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1))
            val_env.reset()
            
            while not val_env.is_over:
                state = val_env.game.state
                legal = state.getLegalPacmanActions()
                if Directions.STOP in legal:
                    legal.remove(Directions.STOP)
                if not legal:
                    break
                
                agent.update_position_buffer(state)
                with torch.no_grad():
                    probs = net(agent.state_to_tensor(state), return_both=False).squeeze()
                mask = torch.tensor([1.0 if a in legal else 0.0 for a in agent.actions])
                masked = probs * mask
                masked = masked / masked.sum() if masked.sum() > 0 else mask / mask.sum()
                action = agent.actions[torch.multinomial(masked, 1).item()]
                val_env.step(action)
            
            score = val_env.game.state.getScore()
            won = val_env.game.state.isWin()
            tqdm.write(f"\nEpoch {epoch+1}: Score={score} {'WON!' if won else ''}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Wins: {wins}/{num_epochs*batch_size} ({100*wins/(num_epochs*batch_size):.1f}%)")
    
    torch.save(net.state_dict(), 'reinforcement_learning/actor_critic_trained.pth')
    print("Model saved!")
    
    return net


if __name__ == "__main__":
    VALIDATE = False
    
    if VALIDATE:
        net = ActorCriticNetwork(memory_context=5)
        net.load_state_dict(torch.load('reinforcement_learning/actor_critic_trained.pth'), strict=False)
        net.eval()
        
        agent = RLAgent(memory_context=5)
        agent.model = net
        
        env = PacmanEnv(agent, 'mediumClassic', graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1))
        env.reset()
        
        while not env.is_over:
            state = env.game.state
            legal = state.getLegalPacmanActions()
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)
            if not legal:
                break
            
            agent.update_position_buffer(state)
            with torch.no_grad():
                probs = net(agent.state_to_tensor(state), return_both=False).squeeze()
            mask = torch.tensor([1.0 if a in legal else 0.0 for a in agent.actions])
            masked = probs * mask
            masked = masked / masked.sum() if masked.sum() > 0 else mask / mask.sum()
            action = agent.actions[torch.multinomial(masked, 1).item()]
            env.step(action)
        
        print(f"Score: {env.game.state.getScore()}, Won: {env.game.state.isWin()}")
    else:
        train(num_epochs=100, batch_size=32, steps_per_epoch=20, 
              layout_name='mediumClassic', gamma=0.95, lam=0.95, lr=1e-5, show_epochs=50)
