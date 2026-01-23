"""
DQN Training Script for Pacman
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning.dqn_agent import DQNAgent, Directions
from reinforcement_learning.dqn_model import DQN
import core.layout as layout_module
from core.pacman import ClassicGameRules
import display.textDisplay as textDisplay
import agents.ghostAgents as ghostAgents
import display.graphicsDisplay as graphicsDisplay

def run_validation_game(agent, layout_name='mediumClassic'):
    """
    Run a single validation game with graphics to show progress.
    """
    # Save current epsilon
    old_epsilon = agent.epsilon
    agent.epsilon = 0.05 # Low epsilon for validation (mostly exploitation)
    
    lay = layout_module.getLayout(layout_name)
    ghosts = [ghostAgents.RandomGhost(i+1) for i in range(4)]
    
    # Initialize graphics
    display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1)
    
    rules = ClassicGameRules()
    game = rules.newGame(lay, agent, ghosts, display, quiet=False, catchExceptions=False)
    
    # Run game
    game.run()
    
    # Restore epsilon
    agent.epsilon = old_epsilon
    
    score = game.state.getScore()
    won = game.state.isWin()
    return score, won

def run_dqn_training(num_epochs=100, max_steps_per_game=500, update_target_every=10, layout_name='mediumClassic', show_every=10):
    
    print("Initializing DQN Training...")
    
    agent = DQNAgent(epsilon=1.0)
    optimizer = optim.Adam(agent.model.parameters(), lr=1e-4)
    loss_fn = torch.nn.SmoothL1Loss() # Huber Loss
    
    # Load Layout
    layout = layout_module.getLayout(layout_name)
    ghosts = [ghostAgents.RandomGhost(i+1) for i in range(4)]
    rules = ClassicGameRules()
    
    total_wins = 0
    scores = []
    losses = []
    
    pbar = tqdm(range(num_epochs), desc="DQN Training")
    
    for epoch in pbar:
        
        # Start new game
        game = rules.newGame(layout, agent, ghosts, textDisplay.NullGraphics(), quiet=True, catchExceptions=False)
        agent.registerInitialState(game.state)
        
        state_tensor = agent.get_temporal_input(game.state)
        
        total_reward = 0
        steps = 0
        
        while not game.gameOver and steps < max_steps_per_game:
            steps += 1
            
            # 1. Select Action (Epsilon-Greedy inside agent.getAction)
            # We need to manually separate exploration vs exploitation logic if we want to store transitions perfectly
            # But agent.getAction() handles the state buffer.
            # To get specific action index, we need to map back.
            
            action = agent.getAction(game.state)
            action_idx = agent.action_to_idx.get(action, 0) # Default to 0 if stop/unknown
            
            prev_score = game.state.getScore()
            prev_state_tensor = state_tensor.clone() # Store S
            
            # 2. Step Environment
            game.state = game.state.generateSuccessor(0, action)
            game.display.update(game.state.data)
            game.rules.process(game.state, game)
            
            # 3. Get Reward & Next State
            new_score = game.state.getScore()
            reward = new_score - prev_score
            
            done = game.gameOver
            if done and game.state.isWin():
                reward += 100 # Win Bonus
                total_wins += 1
            
            if done:
                next_state_tensor = None
            else:
                next_state_tensor = agent.get_temporal_input(game.state) # S'
                state_tensor = next_state_tensor
                
            # 4. Push to Replay Buffer
            # (S, A, R, S', Done)
            # Store tensors directly to save conversion time, or numpy to save VRAM? 
            # Storing tensors on CPU is safer for memory.
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
                
                batch_state = torch.cat(batch_state).to(agent.model.conv_input.weight.device) # (B, 5, 5, H, W)
                batch_action = torch.tensor(batch_action).unsqueeze(1).to(agent.model.conv_input.weight.device) # (B, 1)
                batch_reward = torch.tensor(batch_reward).float().unsqueeze(1).to(agent.model.conv_input.weight.device) # (B, 1)
                batch_done = torch.tensor(batch_done).float().unsqueeze(1).to(agent.model.conv_input.weight.device) # (B, 1)
                
                # Compute Q(s, a)
                q_values = agent.model(batch_state) # (B, 4)
                current_q = q_values.gather(1, batch_action) # (B, 1)
                
                # Compute Target Q
                # Q_target = r + gamma * max(Q_target_net(s', a'))
                # Handle next states
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])
                
                next_q_values = torch.zeros(agent.batch_size).to(agent.model.conv_input.weight.device)
                with torch.no_grad():
                     if len(non_final_next_states) > 0:
                        next_q_values[non_final_mask] = agent.target_model(non_final_next_states).max(1)[0]
                
                target_q = batch_reward + (0.99 * next_q_values.unsqueeze(1)) * (1 - batch_done)
                
                # Loss
                loss = loss_fn(current_q, target_q)
                losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
                optimizer.step()
                
        # End of Game/Epoch
        agent.decay_epsilon()
        scores.append(game.state.getScore())
        
        # Update Target Network
        if epoch % update_target_every == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            
        # Logging
        avg_score = np.mean(scores[-10:])
        avg_loss = np.mean(losses[-100:]) if losses else 0
        pbar.set_postfix({'Score': avg_score, 'Eps': agent.epsilon, 'Wins': total_wins, 'Loss': avg_loss})
    
        # Visualization / Validation
        if (epoch + 1) % show_every == 0:
            tqdm.write(f"\nRunning Validation Game at Epoch {epoch+1}...")
            v_score, v_won = run_validation_game(agent, layout_name)
            tqdm.write(f"Validation Result: Score={v_score}, Won={v_won}")

        # Save Model
        if epoch % 50 == 0:
            torch.save(agent.model.state_dict(), 'reinforcement_learning/dqn_trained.pth')

    print("Training Complete. Saving model...")
    torch.save(agent.model.state_dict(), 'reinforcement_learning/dqn_trained.pth')

if __name__ == '__main__':
    run_dqn_training(num_epochs=1000)
