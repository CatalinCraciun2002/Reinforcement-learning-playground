"""
Pacman RL Environment Wrapper

Simplified environment interface for training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import layout
from core.pacman import ClassicGameRules
from display import textDisplay
from agents import ghostAgents


class PacmanEnv:
    """Lightweight Pacman environment for RL training."""
    
    def __init__(self, agent, layout_name='mediumClassic', display=None):
        self.agent = agent
        self.layout_name = layout_name
        self.display = display or textDisplay.NullGraphics()
        self.game = None
        self.reset()
    
    def reset(self):
        """Reset environment and return initial state."""
        lay = layout.getLayout(self.layout_name)
        ghosts = [ghostAgents.RandomGhost(i+1) for i in range(4)]
        rules = ClassicGameRules()
        self.game = rules.newGame(lay, self.agent, ghosts, self.display, quiet=True, catchExceptions=False)
        self.agent.registerInitialState(self.game.state)
        return self.game.state.deepCopy()
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)."""
        prev_score = self.game.state.getScore()
        
        try:
            self.game.state = self.game.state.generateSuccessor(0, action)
            self.game.display.update(self.game.state.data)
            self.game.rules.process(self.game.state, self.game)
        except:
            return self.game.state.deepCopy(), -500, True
        
        reward = self.game.state.getScore() - prev_score
        done = self.game.gameOver
        
        # Win bonus
        if done and self.game.state.isWin():
            reward += 500
            
        # Proximity Penalty to encourage faster reaction
        if not done:
            ghost_pos = self.game.state.getGhostPositions()
            if ghost_pos:
                pacman_pos = self.game.state.getPacmanPosition()
                min_dist = min([abs(pacman_pos[0]-g[0]) + abs(pacman_pos[1]-g[1]) for g in ghost_pos])
                if min_dist <= 1: reward -= 50
                elif min_dist <= 2: reward -= 20
                elif min_dist <= 3: reward -= 10
        
        return self.game.state.deepCopy(), reward, done
    
    @property
    def is_over(self):
        return self.game.gameOver
