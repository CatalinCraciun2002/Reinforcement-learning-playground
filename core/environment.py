"""
Pacman RL Environment Wrapper

Simplified environment interface for training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

from core import layout
from core.pacman import ClassicGameRules
from display import textDisplay
from agents.base_agents import ghostAgents
from core.game import Directions

# Reward Constants - All rewards/penalties defined here
REWARD_FOOD = 10          # Reward for eating a food pellet
REWARD_CAPSULE = 50       # Reward for eating a power capsule
REWARD_GHOST = 200        # Reward for eating a scared ghost
REWARD_WIN = 1000          # Reward for winning (eating all food)
PENALTY_DEATH = -1000      # Penalty for being killed by a ghost
PENALTY_TIME = -1        # Penalty per step (encourages faster play)

# Reward scaling to normalize to [-1, 1] range
REWARD_SCALE = 1000.0     # Maximum absolute reward value


class PacmanEnv:
    """Lightweight Pacman environment for RL training."""
    
    def __init__(self, agent, layout_name='mediumClassic', add_extra_ghost=False, display=None, allow_stop=True, env_id=0):
        self.agent = agent
        self.layout_name = layout_name
        self.display = display or textDisplay.NullGraphics()
        self.game = None
        self.wins = 0
        self.allow_stop = allow_stop
        self.add_extra_ghost = add_extra_ghost
        self.env_id = env_id
        self.reset()
    
    def reset(self):
        """Reset environment and return initial state."""
        lay = layout.getLayout(self.layout_name)
        ghosts = [ghostAgents.DirectionalGhost(1, 0.8, 0.8)] + [ghostAgents.RandomGhost(i+1) for i in range(1, 4)]
        rules = ClassicGameRules()
        self.game = rules.newGame(lay, self.agent, ghosts, self.display, quiet=True, catchExceptions=False)
        self.agent.registerInitialState(self.game.state, self.env_id)
        
        # Initialize display for graphics
        self.display.initialize(self.game.state.data)
        
        # Add extra ghost after display is initialized
        if self.add_extra_ghost:
            self.add_to_layout_ghost()
        
        return self.game.state # Return the initial state
    
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
        
        # Scale reward to [-1, 1] range
        scaled_reward = reward / REWARD_SCALE
        
        return self.game.state, scaled_reward, done
    
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
    
    def add_directional_ghost(self, position, prob_attack=0.8, prob_scaredFlee=0.8):
        """
        Add a DirectionalGhost agent at the specified position.
        
        Args:
            position: (x, y) tuple for ghost position
            prob_attack: Probability of attacking when not scared
            prob_scaredFlee: Probability of fleeing when scared
        """
        from agents.base_agents.ghostAgents import DirectionalGhost
        from core.game import AgentState, Configuration
        
        # Create new ghost agent with next available index
        ghost_index = len(self.game.agents)
        ghost_agent = DirectionalGhost(ghost_index, prob_attack, prob_scaredFlee)
        
        # Add ghost agent to the game
        self.game.agents.append(ghost_agent)
        
        # Create ghost state and add to game state
        ghost_state = AgentState(Configuration(position, Directions.STOP), False)
        self.game.state.data.agentStates.append(ghost_state)
        
        # If display has been initialized, add the ghost image
        if hasattr(self.game.display, 'agentImages'):
            ghost_image = self.game.display.drawGhost(ghost_state, ghost_index)
            self.game.display.agentImages.append((ghost_state, ghost_image))



    def add_to_layout_ghost(self):
        # Add DirectionalGhost in the corridor where Pacman starts (on random side)
        
        pacman_pos = self.game.state.getPacmanPosition()
        
        # Place ghost on random side (left or right) of Pacman
        # Assumes corridor is horizontal - adjust x coordinate
        side = random.choice([-1, 1])  # -1 for left, 1 for right
        
        # Find a valid position in the corridor (3 units away from pacman)
        ghost_x = int(pacman_pos[0] + side * 3)
        ghost_y = int(pacman_pos[1])
        ghost_pos = (ghost_x, ghost_y)
        
        # Add the DirectionalGhost
        self.add_directional_ghost(ghost_pos, prob_attack=1, prob_scaredFlee=0)

            
            
            