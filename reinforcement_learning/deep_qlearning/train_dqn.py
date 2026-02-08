"""
DQN Training Script for Pacman
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning.deep_qlearning.dqn_agent import DQNAgent, Directions
from reinforcement_learning.dqn_model import DQN
import core.layout as layout_module
from core.pacman import ClassicGameRules
import display.textDisplay as textDisplay
import agents.ghostAgents as ghostAgents
import display.graphicsDisplay as graphicsDisplay

def run_validation_game(agent, layout_name='mediumClassic', view_speed=0.1):
    """
    Run a single validation game with graphics to show progress.
    """
    # Save current epsilon
    old_epsilon = agent.epsilon
    agent.epsilon = 0.05 # Low epsilon for validation (mostly exploitation)
    
    lay = layout_module.getLayout(layout_name)
    ghosts = [ghostAgents.RandomGhost(i+1) for i in range(4)]
    
    # Initialize graphics
    display = graphicsDisplay.PacmanGraphics(1.0, frameTime=view_speed)
    
    rules = ClassicGameRules()
    game = rules.newGame(lay, agent, ghosts, display, quiet=False, catchExceptions=False)
    
    # Run game
    game.run()
    
    # Restore epsilon
    agent.epsilon = old_epsilon
    
    score = game.state.getScore()
    won = game.state.isWin()
    return score, won

def run_dqn_training(args):
    
    print(f"Initializing DQN Training (Layout: {args.layout})...")
    
    agent = DQNAgent(epsilon=1.0)
    optimizer = optim.Adam(agent.model.parameters(), lr=1e-4)
    loss_fn = torch.nn.SmoothL1Loss() # Huber Loss
    
    # Load Layout
    layout = layout_module.getLayout(args.layout)
    ghosts = [ghostAgents.RandomGhost(i+1) for i in range(4)]
    rules = ClassicGameRules()
    
    total_wins = 0
    scores = []
    losses = []
    
    pbar = tqdm(range(args.epochs), desc="DQN Training")
    
    for epoch in pbar:
        
        # Start new game
        # Use NullGraphics by default for speed, unless rendering this specific episode
        if args.render_every > 0 and (epoch + 1) % args.render_every == 0:
             # Just run validation game separate from training to avoid messing up training loop speed?
             # Or we can run THIS training episode with graphics.
             # Let's run a separate validation game to be clean and safe, 
             # as the training loop is optimized for speed (NullGraphics)
             pass 

        game = rules.newGame(layout, agent, ghosts, textDisplay.NullGraphics(), quiet=True, catchExceptions=False)
        agent.registerInitialState(game.state)
        
        state_tensor = agent.get_temporal_input(game.state)
        
        total_reward = 0
        steps = 0
        episode_losses = []
        episode_q = []
        
        while not game.gameOver and steps < args.max_steps:
            steps += 1
            
            # 1. Select Action
            action = agent.getAction(game.state)
            action_idx = agent.action_to_idx.get(action, 0)
            
            prev_score = game.state.getScore()
            prev_state_tensor = state_tensor.clone()
            
            # 2. Step Environment
            game.state = game.state.generateSuccessor(0, action)
            game.display.update(game.state.data) # Null display just passes
            game.rules.process(game.state, game)
            
            # 3. Get Reward & Next State
            new_score = game.state.getScore()
            reward = new_score - prev_score
            
            done = game.gameOver
            if done and game.state.isWin():
                reward += 100 # Win Bonus
                total_wins += 1
            
            if action != Directions.STOP:
                reward += 0.1
                
            # Living Penalty (optional, to encourage shorter paths if score is same)
            # reward -= 1 
            
            if done:
                next_state_tensor = None
            else:
                next_state_tensor = agent.get_temporal_input(game.state)
                state_tensor = next_state_tensor
                
            # 4. Push to Replay Buffer
            agent.memory.push(
                prev_state_tensor.cpu(), 
                action_idx, 
                reward, 
                next_state_tensor.cpu() if next_state_tensor is not None else None, 
                done
            )
            
            total_reward += reward
            
            # 5. Train
            if len(agent.memory) > agent.batch_size:
                transitions = agent.memory.sample(agent.batch_size)
                # Transpose batch
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                
                # Batches are already (1, 25, H, W), so cat makes (B, 25, H, W)
                batch_state = torch.cat(batch_state).to(agent.model.conv1.weight.device)
                batch_action = torch.tensor(batch_action).unsqueeze(1).to(agent.model.conv1.weight.device)
                batch_reward = torch.tensor(batch_reward).float().unsqueeze(1).to(agent.model.conv1.weight.device)
                batch_done = torch.tensor(batch_done).float().unsqueeze(1).to(agent.model.conv1.weight.device)
                
                # Compute Q(s, a)
                q_values = agent.model(batch_state) # (B, 4)
                current_q = q_values.gather(1, batch_action) # (B, 1)
                episode_q.append(current_q.mean().item())
                
                # Compute Target Q
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])
                
                next_q_values = torch.zeros(agent.batch_size).to(agent.model.conv1.weight.device)
                with torch.no_grad():
                     if len(non_final_next_states) > 0:
                        # Double DQN or Standard? Standard for now: max(Q_target)
                        next_q_values[non_final_mask] = agent.target_model(non_final_next_states).max(1)[0]
                
                target_q = batch_reward + (0.99 * next_q_values.unsqueeze(1)) * (1 - batch_done)
                
                # Loss
                loss = loss_fn(current_q, target_q)
                episode_losses.append(loss.item())
                losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
                optimizer.step()
                
        # End of Game/Epoch
        agent.decay_epsilon()
        final_score = game.state.getScore()
        scores.append(final_score)
        
        # Update Target Network
        if epoch % args.update_target_every == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            
        # Logging
        avg_score = np.mean(scores[-100:])
        avg_loss = np.mean(losses[-100:]) if losses else 0
        avg_q = np.mean(episode_q) if episode_q else 0
        
        # Detailed log per episode if verbose
        if args.verbose:
            tqdm.write(f"Ep {epoch+1}: Score={final_score}, Steps={steps}, AvgQ={avg_q:.2f}, Loss={np.mean(episode_losses) if episode_losses else 0:.4f}, Eps={agent.epsilon:.2f}")

        pbar.set_postfix({'Score': f"{avg_score:.1f}", 'Wins': total_wins, 'Q': f"{avg_q:.1f}", 'L': f"{avg_loss:.3f}"})
    
        # Visualization / Validation
        if args.render_every > 0 and (epoch + 1) % args.render_every == 0:
            tqdm.write(f"\nRunning Validation Game at Epoch {epoch+1}...")
            v_score, v_won = run_validation_game(agent, args.layout, args.view_speed)
            tqdm.write(f"Validation Result: Score={v_score}, Won={v_won}\n")

        # Save Model
        if (epoch+1) % 50 == 0:
            torch.save(agent.model.state_dict(), 'reinforcement_learning/dqn_trained.pth')

    print("Training Complete. Saving model...")
    torch.save(agent.model.state_dict(), 'reinforcement_learning/dqn_trained.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Pacman Training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Layout name')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--update_target_every', type=int, default=10, help='Update target network every N episodes')
    parser.add_argument('--render_every', type=int, default=10, help='Render validation game every N episodes (0 to disable)')
    parser.add_argument('--verbose', action='store_true', help='Print stats for every episode')
    parser.add_argument('--view_speed', type=float, default=0.05, help='Speed of rendered game')
    
    args = parser.parse_args()
    
    run_dqn_training(args)
