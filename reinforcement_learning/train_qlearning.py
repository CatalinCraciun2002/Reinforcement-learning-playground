
"""
Training script for Approximate Q-Learning Agent
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import pickle
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning.qlearning_agent import ApproximateQAgent
import core.layout as layout_module
from core.pacman import ClassicGameRules
import display.textDisplay as textDisplay
import display.graphicsDisplay as graphicsDisplay
import agents.ghostAgents as ghostAgents
from core.game import Directions
from reinforcement_learning.distance_utils import get_path_to_food, get_paths_to_all_scared_ghosts, get_paths_to_all_active_ghosts, get_path_to_capsules
from reinforcement_learning.plot_utils import InGameTrajectoryVisualizer

def run_qlearning_training(args):
    print(f"Initializing Q-Learning Training (Layout: {args.layout})...")
    
    # Initialize Agent
    agent = ApproximateQAgent(
        alpha=args.alpha, 
        gamma=args.gamma, 
        epsilon=args.epsilon, 
        numTraining=args.episodes
    )
    
    # Load Weights if exists
    if os.path.exists(args.weights_file):
        print(f"Loading weights from {args.weights_file}...")
        with open(args.weights_file, 'rb') as f:
            agent.weights = pickle.load(f)

    layout = layout_module.getLayout(args.layout)
    ghosts = [ghostAgents.DirectionalGhost(i+1) for i in range(4)]
    rules = ClassicGameRules()
    
    scores = []
    wins = 0
    
    pbar = tqdm(range(args.episodes), desc="Training")
    visualizer = None
    
    for episode in pbar:
        # Check if we should render this episode
        render = (args.render_every > 0 and (episode + 1) % args.render_every == 0)
        
        if render:
            display = graphicsDisplay.PacmanGraphics(1.0, frameTime=args.view_speed)
        else:
            display = textDisplay.NullGraphics()
            
        game = rules.newGame(layout, agent, ghosts, display, quiet=True, catchExceptions=False)
        game.display.initialize(game.state.data)
        
        # Initialize GameVisualizer with screen dimensions
        if render:
            if visualizer is None:
                visualizer = InGameTrajectoryVisualizer(game.display)
            else:
                visualizer.reset()
        # But ClassicGameRules.run() calls agent.getAction which calls update implicitly?
        # No, standard Pacman agents don't update in getAction usually.
        # But our ApproximateQAgent needs (state, action, nextState, reward) to update.
        # The standard game loop in game.py might not support reinforcement learning updates easily
        # unless we hook into it or run it step-by-step.
        
        # Let's run step-by-step
        step_count = 0
        episode_start_time = time.time()
        while not game.gameOver:
            for i, agent_obj in enumerate(game.agents):
                if game.gameOver: break
                
                # 1. Get Action
                state = game.state
                action = agent_obj.getAction(state)
                
                # 2. Execute Action
                game.state = game.state.generateSuccessor(i, action)
                game.display.update(game.state.data)
                game.rules.process(game.state, game)
                
                # 3. If it was Pacman's turn, update the learning agent
                if i == 0:
                    step_count += 1
                    next_state = game.state
                    reward = next_state.getScore() - state.getScore()
                    
                    # Step penalty to encourage speed and discourage idling
                    reward -= 1
                    
                    # Time penalty based on wall-clock time
                    current_time = time.time()
                    elapsed_time = current_time - episode_start_time
                    # Penalize more for longer execution (e.g., -10 points per second)
                    reward -= (elapsed_time * 0.1) 
                    
                    # Win/Lose bonus
                    if game.gameOver:
                        if game.state.isWin():
                            # Large gain if finished fast - increased for more aggressive speed incentive
                            speed_bonus = max(0, 2000 - step_count * 5)
                            
                            # Additional real-time bonus
                            time_bonus = max(0, 1000 - elapsed_time * 20)
                            reward += 500 + speed_bonus + time_bonus
                            wins += 1
                        elif game.state.isLose():
                            reward -= 500
                    
                    # Proximity Penalty to encourage faster reaction
                    # ONLY apply to active ghosts
                    active_ghost_pos = [g.getPosition() for g in next_state.getGhostStates() if g.scaredTimer == 0]
                    if active_ghost_pos:
                        min_dist = min([abs(next_state.getPacmanPosition()[0]-g[0]) + abs(next_state.getPacmanPosition()[1]-g[1]) for g in active_ghost_pos])
                        if min_dist <= 1: reward -= 50
                        elif min_dist <= 2: reward -= 20
                        elif min_dist <= 3: reward -= 10
                        
                    # Proximity Bonus for scared ghosts to encourage chasing
                    scared_ghost_pos = [g.getPosition() for g in next_state.getGhostStates() if g.scaredTimer > 0]
                    if scared_ghost_pos:
                        min_scared_dist = min([abs(next_state.getPacmanPosition()[0]-g[0]) + abs(next_state.getPacmanPosition()[1]-g[1]) for g in scared_ghost_pos])
                        if min_scared_dist <= 1: reward += 30
                        elif min_scared_dist <= 2: reward += 15
                        elif min_scared_dist <= 3: reward += 10
                    
                    # Eat Ghost Bonus
                    if next_state.getScore() - state.getScore() > 100:
                        reward += 200 # Significant bonus for eating a ghost
                    
                    # Update Agent
                    agent.update(state, action, next_state, reward)
                    
                    # Update Visualization if rendering
                    if render and visualizer:
                        g_paths = get_paths_to_all_active_ghosts(next_state)
                        f_path = get_path_to_food(next_state)
                        s_paths = get_paths_to_all_scared_ghosts(next_state)
                        c_path = get_path_to_capsules(next_state)
                        visualizer.update(g_paths, c_path, f_path, s_paths, next_state)
        
        # End of episode
        # agent.final(game.state) # Optional, usually for decay
        
        final_score = game.state.getScore()
        scores.append(final_score)
        
        # Decay epsilon
        if agent.epsilon > 0.05:
            agent.epsilon *= 0.99
            
        # Logging
        avg_score = sum(scores[-100:]) / min(len(scores), 100)
        pbar.set_postfix({'AvgScore': f"{avg_score:.1f}", 'Wins': wins, 'Eps': f"{agent.epsilon:.2f}"})

    print("\nTraining Complete.")
    print(f"Final Weights: {agent.weights}")
    
    # Save weights
    with open(args.weights_file, 'wb') as f:
        pickle.dump(agent.weights, f)
    print(f"Weights saved to {args.weights_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-Learning Pacman Training')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Layout name')
    parser.add_argument('--render_every', type=int, default=10, help='Render every N episodes')
    parser.add_argument('--view_speed', type=float, default=0.05, help='Speed of rendered game')
    # Hyperparameters
    parser.add_argument('--alpha', type=float, default=0.2, help='Learning Rate')
    parser.add_argument('--gamma', type=float, default=0.8, help='Discount Factor')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Exploration Rate')
    parser.add_argument('--weights_file', type=str, default='agents/qlearning_weights.pkl', help='File to save/load weights')

    args = parser.parse_args()
    run_qlearning_training(args)
