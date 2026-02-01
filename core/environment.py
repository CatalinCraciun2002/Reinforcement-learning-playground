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
from core.game import Directions

# Reward Constants - All rewards/penalties defined here
REWARD_FOOD = 10          # Reward for eating a food pellet
REWARD_CAPSULE = 50       # Reward for eating a power capsule
REWARD_GHOST = 200        # Reward for eating a scared ghost
REWARD_WIN = 500          # Reward for winning (eating all food)
PENALTY_DEATH = -500      # Penalty for being killed by a ghost
PENALTY_TIME = -1         # Penalty per step (encourages faster play)


class PacmanEnv:
    """Lightweight Pacman environment for RL training."""
    
    def __init__(self, agent, layout_name='mediumClassic', display=None, allow_stop=True):
        self.agent = agent
        self.layout_name = layout_name
        self.display = display or textDisplay.NullGraphics()
        self.game = None
        self.wins = 0
        self.allow_stop = allow_stop
        self.reset()
    
    def reset(self):
        """Reset environment and return initial state."""
        lay = layout.getLayout(self.layout_name)
        ghosts = [ghostAgents.RandomGhost(i+1) for i in range(4)]
        rules = ClassicGameRules()
        self.game = rules.newGame(lay, self.agent, ghosts, self.display, quiet=True, catchExceptions=False)
        self.agent.registerInitialState(self.game.state)
        
        # Initialize display for graphics
        self.display.initialize(self.game.state.data)
        
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)."""
        # Track state before action
        prev_food_count = self.game.state.getNumFood()
        prev_capsules = len(self.game.state.getCapsules())
        prev_ghost_states = [g.scaredTimer for g in self.game.state.getGhostStates()]
        
        # Move Pacman (agent 0)
        self.game.state = self.game.state.generateSuccessor(0, action)
        self.game.display.update(self.game.state.data)
        
        # Move ghosts (agents 1-4)
        for i in range(1, len(self.game.agents)):
            if self.game.state.isWin() or self.game.state.isLose():
                break
            ghost_action = self.game.agents[i].getAction(self.game.state)
            self.game.state = self.game.state.generateSuccessor(i, ghost_action)
            self.game.display.update(self.game.state.data)
        
        # Check game over conditions
        self.game.rules.process(self.game.state, self.game)
        
        # Calculate reward based on state changes
        reward = 0
        done = self.game.gameOver
        
        # Time penalty per step
        reward += PENALTY_TIME
        
        # Reward for eating food
        curr_food_count = self.game.state.getNumFood()
        food_eaten = prev_food_count - curr_food_count
        reward += food_eaten * REWARD_FOOD
        
        # Reward for eating capsules
        curr_capsules = len(self.game.state.getCapsules())
        capsules_eaten = prev_capsules - curr_capsules
        reward += capsules_eaten * REWARD_CAPSULE
        
        # Reward for eating scared ghosts
        curr_ghost_states = [g.scaredTimer for g in self.game.state.getGhostStates()]
        for prev_scared, curr_scared in zip(prev_ghost_states, curr_ghost_states):
            # Ghost was scared before, now reset (timer went to 0 suddenly) = eaten
            if prev_scared > 0 and curr_scared == 0:
                # Check if ghost position was reset (indicates it was eaten, not timer expired)
                reward += REWARD_GHOST
        
        # Check win/loss conditions
        if done:
            if self.game.state.isWin():
                reward += REWARD_WIN
                self.wins += 1
            elif self.game.state.isLose():
                reward += PENALTY_DEATH
        
        return self.game.state, reward, done
    
    @property
    def is_over(self):
        return self.game.gameOver

    def get_legal(self, state):

        if self.allow_stop:
            return state.getLegalPacmanActions()

        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        return legal
