"""
Approximate Q-Learning Agent for Pacman
Uses linear feature approximation.
"""

import random, time
import core.util as util
import sys
import os
import math

# Add parent directory to path to import game module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.game import Agent, Directions, Actions

class SimpleExtractor:
    """
    Returns features for Pacman:
    - bias: 1.0
    - active-ghosts: # of dangerous ghosts 1 step away
    - eats-food: 1.0 if pacman eats food
    - closest-food: distance to closest food
    - scared-ghosts: # of scared ghosts 1 step away
    - closest-scared-ghost: distance to closest scared ghost
    """
    def getFeatures(self, state, action):
        features = util.Counter()
        features['bias'] = 1.0
        
        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # Distinguish between active and scared ghosts
        active_ghosts = []
        scared_ghosts = []
        for ghost in state.getGhostStates():
            if ghost.scaredTimer > 0:
                scared_ghosts.append(ghost.getPosition())
            else:
                active_ghosts.append(ghost.getPosition())
        
        # Count dangerous ghosts 1-step away
        features['#-of-active-ghosts-1-step-away'] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, state.getWalls()) 
            for g in active_ghosts
        )

        # Count scared ghosts 1-step away (opportunistic eating)
        features['#-of-scared-ghosts-1-step-away'] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, state.getWalls()) 
            for g in scared_ghosts
        )

        # if there is no danger of ghosts then add the food feature
        if not features['#-of-active-ghosts-1-step-away']:
            if state.getFood()[next_x][next_y]:
                features['eats-food'] = 1.0
            
        # Food distance
        food_dist = closestTarget((next_x, next_y), state.getFood(), state.getWalls())
        if food_dist is not None:
            features['closest-food'] = float(food_dist) / (state.getWalls().width * state.getWalls().height)

        # Scared ghost distance
        if scared_ghosts:
            scared_dist = closestTarget((next_x, next_y), scared_ghosts, state.getWalls(), is_grid=False)
            if scared_dist is not None:
                features['closest-scared-ghost'] = float(scared_dist) / (state.getWalls().width * state.getWalls().height)
            
        return features

def closestTarget(pos, target, walls, is_grid=True):
    """
    BFS to find distance to closest target (can be a Grid/matrix or a list of positions).
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        
        # Check if we hit a target
        if is_grid:
            if target[pos_x][pos_y]: return dist
        else:
            if (pos_x, pos_y) in target: return dist
            
        # Spread out
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    return None

class ApproximateQAgent(Agent):
    """
    ApproximateQLearningAgent

    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """
    def __init__(self, extractor='SimpleExtractor', alpha=0.2, gamma=0.8, epsilon=0.05, numTraining=100, **args):
        self.featExtractor = SimpleExtractor() # simplified instantiation
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.weights = util.Counter()
        self.index = 0 # Pacman is always agent 0

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        return features * self.weights

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        diff = (reward + self.gamma * self.getValue(nextState)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        
        for feature, value in features.items():
            self.weights[feature] += self.alpha * diff * value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
        
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        return max([self.getQValue(state, action) for action in actions])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None
        
        best_val = float('-inf')
        best_actions = []
        
        for action in actions:
            q_val = self.getQValue(state, action)
            if q_val > best_val:
                best_val = q_val
                best_actions = [action]
            elif q_val == best_val:
                best_actions.append(action)
        
        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
            
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        
        return self.getPolicy(state)
        
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        self.episodesSoFar += 1
        if self.episodesSoFar % 100 == 0:
             print(f"Episodes completed: {self.episodesSoFar}")

    def getLegalActions(self, state):
        return state.getLegalPacmanActions()
    
    def registerInitialState(self, state):
        pass
