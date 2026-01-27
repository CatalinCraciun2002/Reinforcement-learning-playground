
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
    
    for episode in pbar:
        # Check if we should render this episode
        render = (args.render_every > 0 and (episode + 1) % args.render_every == 0)
        
        if render:
            display = graphicsDisplay.PacmanGraphics(1.0, frameTime=args.view_speed)
        else:
            display = textDisplay.NullGraphics()
            
        game = rules.newGame(layout, agent, ghosts, display, quiet=True, catchExceptions=False)
        game.display.initialize(game.state.data)
        
        # Run the game loop manually to allow for agent updates
        # But ClassicGameRules.run() calls agent.getAction which calls update implicitly?
        # No, standard Pacman agents don't update in getAction usually.
        # But our ApproximateQAgent needs (state, action, nextState, reward) to update.
        # The standard game loop in game.py might not support reinforcement learning updates easily
        # unless we hook into it or run it step-by-step.
        
        # Let's run step-by-step
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
                    next_state = game.state
                    reward = next_state.getScore() - state.getScore()
                    
                    # Win/Lose bonus
                    if game.gameOver:
                        if game.state.isWin():
                            reward += 500
                            wins += 1
                        elif game.state.isLose():
                            reward -= 500
                    
                    # Update Agent
                    if action != Directions.STOP:
                        reward += 0
                    agent.update(state, action, next_state, reward)
        
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
