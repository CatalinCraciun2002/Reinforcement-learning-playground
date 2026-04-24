"""
Record Human Gameplay for Training Data Collection

This script allows you to play Pacman with keyboard controls while
recording all game states, actions, and rewards for later use in training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import pickle
import numpy as np

from core import layout
from core.pacman import ClassicGameRules
from agents.base_agents import ghostAgents, keyboardAgents
from display import graphicsDisplay
# Default reward constants for human recordings
REWARD_FOOD = 10
REWARD_CAPSULE = 50
REWARD_GHOST = 200
REWARD_WIN = 1000
PENALTY_DEATH = -1000
PENALTY_TIME = -1


class RecordingEnvironment:
    """Environment wrapper that records all transitions during gameplay."""
    
    def __init__(self, layout_name='mediumClassic', zoom=1.0, min_score_threshold=None):
        self.layout_name = layout_name
        self.zoom = zoom
        self.min_score_threshold = min_score_threshold
        self.transitions = []
        self.game = None
        
    def play_and_record(self):
        """Start a keyboard-controlled game and record all transitions."""
        print("Starting Pacman with keyboard controls...")
        print("Controls: Arrow keys or WASD to move")
        print("Recording gameplay data...\n")
        
        # Setup game
        lay = layout.getLayout(self.layout_name)
        pacman_agent = keyboardAgents.KeyboardAgent()
        ghosts = [ghostAgents.RandomGhost(i+1) for i in range(lay.getNumGhosts())]
        display = graphicsDisplay.PacmanGraphics(zoom=self.zoom, frameTime=0.1)
        
        rules = ClassicGameRules()
        self.game = rules.newGame(lay, pacman_agent, ghosts, display, 
                                  quiet=False, catchExceptions=False)
        
        # Track initial state
        initial_state = self.game.state.deepCopy()
        
        # Run game and record
        self._run_and_record()
        
        # Save data
        self._save_episode()
        
        return len(self.transitions)
    
    def _run_and_record(self):
        """Run the game and record each transition."""
        self.game.display.initialize(self.game.state.data)
        
        step_num = 0
        while not self.game.gameOver:
            # Save state before action
            prev_state = self._extract_state(self.game.state)
            prev_food_count = self.game.state.getNumFood()
            prev_capsules = len(self.game.state.getCapsules())
            prev_ghost_scared = [g.scaredTimer for g in self.game.state.getGhostStates()]
            
            # Get action from keyboard
            action = self.game.agents[0].getAction(self.game.state)
            
            # Execute Pacman action
            self.game.state = self.game.state.generateSuccessor(0, action)
            self.game.display.update(self.game.state.data)
            
            # Execute ghost actions
            for i in range(1, len(self.game.agents)):
                if self.game.state.isWin() or self.game.state.isLose():
                    break
                ghost_action = self.game.agents[i].getAction(self.game.state)
                self.game.state = self.game.state.generateSuccessor(i, ghost_action)
                self.game.display.update(self.game.state.data)
            
            # Check game over
            self.game.rules.process(self.game.state, self.game)
            
            # Calculate reward (same logic as environment)
            reward = self._calculate_reward(
                prev_food_count, prev_capsules, prev_ghost_scared
            )
            
            # Record transition
            transition = {
                'step': step_num,
                'state': prev_state,
                'action': action,
                'reward': reward,
                'done': self.game.gameOver
            }
            self.transitions.append(transition)
            
            step_num += 1
        
        self.game.display.finish()
    
    def _extract_state(self, game_state):
        """Extract relevant state information in a compact format."""
        return {
            'pacman_pos': game_state.getPacmanPosition(),
            'ghost_positions': game_state.getGhostPositions(),
            'ghost_scared_timers': [g.scaredTimer for g in game_state.getGhostStates()],
            'food_grid': np.array([[game_state.hasFood(x, y) 
                                   for y in range(game_state.data.layout.height)]
                                  for x in range(game_state.data.layout.width)]),
            'capsules': game_state.getCapsules(),
            'walls': None  # Will save once, static
        }
    
    def _calculate_reward(self, prev_food, prev_capsules, prev_ghost_scared):
        """Calculate reward based on state changes (matches environment logic)."""
        reward = PENALTY_TIME
        
        # Food eaten
        curr_food = self.game.state.getNumFood()
        reward += (prev_food - curr_food) * REWARD_FOOD
        
        # Capsules eaten
        curr_capsules = len(self.game.state.getCapsules())
        reward += (prev_capsules - curr_capsules) * REWARD_CAPSULE
        
        # Scared ghosts eaten
        curr_ghost_scared = [g.scaredTimer for g in self.game.state.getGhostStates()]
        for prev, curr in zip(prev_ghost_scared, curr_ghost_scared):
            if prev > 0 and curr == 0:
                reward += REWARD_GHOST
        
        # Win/Loss
        if self.game.gameOver:
            if self.game.state.isWin():
                reward += REWARD_WIN
            elif self.game.state.isLose():
                reward += PENALTY_DEATH
        
        return reward
    
    def _save_episode(self):
        """Save recorded episode to pickle and CSV files (only if score meets threshold)."""
        final_score = self.game.state.getScore()
        outcome = 'WIN' if self.game.state.isWin() else 'LOSS' if self.game.state.isLose() else 'INCOMPLETE'
        
        # Check if score meets minimum threshold
        if self.min_score_threshold is not None and final_score < self.min_score_threshold:
            print(f"\n{'='*60}")
            print(f"Game Over! {outcome}")
            print(f"Final Score: {final_score}")
            print(f"Total Steps: {len(self.transitions)}")
            print(f"Total Reward: {sum(t['reward'] for t in self.transitions):.1f}")
            print(f"\n⚠️  Score {final_score} is below threshold {self.min_score_threshold}")
            print(f"Data NOT saved.")
            print(f"{'='*60}")
            return
        
        # Create output directory if needed
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'game_runs_data'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f'game_{timestamp}_score{int(final_score)}_{outcome}'
        
        # Save walls (static, shared across all states)
        walls = np.array([[self.game.state.hasWall(x, y) 
                          for y in range(self.game.state.data.layout.height)]
                         for x in range(self.game.state.data.layout.width)])
        
        # Create episode data
        episode_data = {
            'layout_name': self.layout_name,
            'transitions': self.transitions,
            'walls': walls,
            'final_score': final_score,
            'outcome': outcome,
            'num_steps': len(self.transitions),
            'reward_constants': {
                'food': REWARD_FOOD,
                'capsule': REWARD_CAPSULE,
                'ghost': REWARD_GHOST,
                'win': REWARD_WIN,
                'death': PENALTY_DEATH,
                'time': PENALTY_TIME
            }
        }
        
        # Save pickle file
        pkl_path = os.path.join(output_dir, f'{base_name}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(episode_data, f)
        
        print(f"\n{'='*60}")
        print(f"Game Over! {outcome}")
        print(f"Final Score: {final_score}")
        print(f"Total Steps: {len(self.transitions)}")
        print(f"Total Reward: {sum(t['reward'] for t in self.transitions):.1f}")
        print(f"\n✓ Data saved to: {pkl_path}")
        print(f"{'='*60}")


# ============================================================================
# CONFIGURATION
# ============================================================================
# Minimum score threshold - only save gameplays with score >= this value
# Set to None to save all gameplays
MIN_SCORE_THRESHOLD = 800  # Change this value to set your desired threshold
# ============================================================================


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Record human Pacman gameplay')
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Layout to play on (default: mediumClassic)')
    parser.add_argument('--zoom', type=float, default=1.0,
                       help='Graphics zoom level (default: 1.0)')
    parser.add_argument('--min-score', type=float, default=MIN_SCORE_THRESHOLD,
                       help=f'Minimum score to save gameplay (default: {MIN_SCORE_THRESHOLD})')
    
    args = parser.parse_args()
    
    env = RecordingEnvironment(
        layout_name=args.layout, 
        zoom=args.zoom,
        min_score_threshold=args.min_score
    )
    num_steps = env.play_and_record()


if __name__ == '__main__':
    main()
