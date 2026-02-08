# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util

class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


def get_closest_food(state):
    agent_x, agent_y = state.getPacmanPosition()
    food_list = state.getFood().asList()
    closest_food = min(food_list, key=lambda food: util.manhattanDistance((agent_x, agent_y), food))
    return closest_food



class CustomAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None


    def getAction(self, state):
        legal_actions = state.getLegalPacmanActions()
        if 'Stop' in legal_actions:
            legal_actions.remove('Stop')

        best_action = None
        best_value = -float('inf')

        pacman_position = state.getPacmanPosition()
        closest_food = get_closest_food(state)

        for action in legal_actions:
            successor_state = state.generatePacmanSuccessor(action)
            successor_position = successor_state.getPacmanPosition()

            ghost_positions = successor_state.getGhostPositions()
            ghost_distances = [util.manhattanDistance(successor_position, ghost_position) for ghost_position in ghost_positions]
            closest_ghost_distance = min(ghost_distances)

            if closest_ghost_distance < 4:
                continue

            food_distance = util.manhattanDistance(successor_position, closest_food)
            value = -food_distance

            if value > best_value:
                best_value = value
                best_action = action

        if best_action is None:
            best_action = legal_actions[0]

        return best_action


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from game import Directions


class FullyConnectedModel(nn.Module):
    def __init__(self, action_size, agents_locations, output_size):
        super(FullyConnectedModel, self).__init__()
        self.fc_a = nn.Linear(action_size, 16)
        self.fc_g = nn.Linear(agents_locations, 16)
        self.fc_c = nn.Linear(4, 16)

        self.fc2 = nn.Linear(48, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, actions, agents_locations, closest_food):

        a = F.relu(self.fc_a(actions))
        g = F.relu(self.fc_g(agents_locations))
        c = F.relu(self.fc_c(closest_food))

        x = torch.cat((a, g, c), dim=1)

        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class NeuralAgent(Agent):
    def __init__(self, ghostNr, useGrid, evalFn="scoreEvaluation"):
        super().__init__()
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

        if useGrid:
            parameters = 220
        else:
            parameters = ghostNr*4

        self.model = FullyConnectedModel(4, parameters, 4)  # 11x20 grid + 4 directions
        self.model.eval()  # Set the model to evaluation mode
        self.use_grid = useGrid

    def gridToEmbedding(self, grid):
        mapping = {'%': 1.0, '.': 0.6, 'o': 0.2, ' ': -0.2, 'P': -0.6, 'G': -1.0}

        embedded = np.array([[mapping[char] for char in row] for row in grid])
        return embedded.flatten()

    def agentsStateToEmbedding(self, agentStates):

        pozitions = []

        for agent in agentStates[1:]:
            pozitions.append(agent.configuration.pos[0] - agentStates[0].configuration.pos[0])
            pozitions.append(agent.configuration.pos[1] - agentStates[0].configuration.pos[1])

        return np.array(pozitions)

    def pozitionsToActions(self, pozitions): # x pozitive = east, x negative = west, y pozitive = north, y negative = south
        actions = [Directions.WEST, Directions.EAST, Directions.NORTH, Directions.SOUTH]
        result = []
        for i in range(0, len(pozitions), 2):
            x = pozitions[i]
            y = pozitions[i + 1]
            if x < 0:
                result.append(0)
                result.append(abs(x))
            else:
                result.append(0)
                result.append(x)

            if y < 0:
                result.append(0)
                result.append(abs(y))
            else:
                result.append(0)
                result.append(y)

        return np.array(result)


    def getAction(self, state):
        # Convert grid to embedding

        if self.use_grid:
            grid = state.data.layout.layoutText
            embedding = self.gridToEmbedding(grid)
        else:
            embedding = self.agentsStateToEmbedding(state.data.agentStates)


        closest_food = get_closest_food(state)


        embedding = self.pozitionsToActions(embedding)
        closest_food = self.pozitionsToActions(closest_food)

        # Check available actions and append 1 if possible, else 0
        legal = state.getLegalPacmanActions()
        actions = [Directions.WEST, Directions.EAST, Directions.NORTH, Directions.SOUTH]
        action_embedding = np.array([1.0 if action in legal else 0.0 for action in actions])

        # Concatenate the embeddings
        embedding_t = torch.from_numpy(embedding).float().unsqueeze(0)
        action_embedding_t = torch.from_numpy(action_embedding).float().unsqueeze(0)
        closest_food_t = torch.from_numpy(closest_food).float().unsqueeze(0)

        # Get the model's output
        with torch.no_grad():
            output = self.model.forward(action_embedding_t, embedding_t, closest_food_t)

        output = output.numpy().flatten()
        output = output * action_embedding

        output_index = np.argmax(output)

        return actions[output_index]


# Agent position 2, ghost position 2* num ghosts, layout height, layout height * layout width = 11 * 20


def scoreEvaluation(state):
    return state.getScore()


