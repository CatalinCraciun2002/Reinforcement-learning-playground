# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class StateAction:

    def __init__(self, position, action, cost, index, parent):
        self.position = position
        self.action = action
        self.cost = cost
        self.parent = parent
        self.index = index

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

import numpy as np


def depthFirstSearch(problem: SearchProblem):


    actions = []
    state_stack = [StateAction(position = problem.getStartState(), action = "Stop", cost = 0, parent = 0, index = 0)]
    grid = np.zeros((1000, 1000))

    while len(state_stack) != 0:

        current_state = state_stack[-1]
        state_stack.pop()
        current_state.index = len(actions)
        actions.append(current_state)

        if problem.isGoalState(current_state.position):
            break

        for successor in problem.getSuccessors(current_state.position):
            if grid[successor[0]] == 0:
                successor_state = StateAction(position = successor[0], action = successor[1],
                                              cost = successor[2], parent = current_state.index, index = 0)
                state_stack.append(successor_state)
                grid[successor[0]] = 1

    final_actions = []

    current_action = actions[-1]

    while current_action.index != 0:
        final_actions.insert(0, current_action.action)
        current_action = actions[current_action.parent]

    return final_actions



def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    actions = []
    state_queue = [[problem.getStartState(), "Stop", 0, 0]]
    grid = np.zeros((1000, 1000))
    visited = []

    while len(state_queue) != 0:

        current_state = state_queue[0]
        del state_queue[0]
        current_state[2] = len(actions)
        actions.append(current_state)

        if problem.isGoalState(current_state[0]):
            break

        for successor in problem.getSuccessors(current_state[0]):
            if isinstance(successor[0], tuple):

                if grid[successor[0]] == 0:
                    l = list(successor)
                    l.append(current_state[2])
                    state_queue.append(l)
                    grid[successor[0]] = 1
            else:
                if successor[0] not in visited:
                    l = list(successor)
                    l.append(current_state[2])
                    state_queue.append(l)
                    visited.append(successor[0])

    final_actions = []

    current_action = actions[-1]

    while current_action[2] != 0:
        final_actions.insert(0, current_action[1])
        current_action = actions[current_action[-1]]

    return final_actions

from queue import PriorityQueue


def uniformCostSearch(problem: SearchProblem):

    """Search the node of least total cost first."""
    actions = []
    state_queue = PriorityQueue()
    state_queue.put((1, [problem.getStartState(), "Stop", 0, 0]))
    grid = np.zeros((1000, 1000))



    while not state_queue.empty():

        current_state = state_queue.get()[1]
        current_state[2] = len(actions)
        actions.append(current_state)

        if problem.isGoalState(current_state[0]):
            break

        for successor in problem.getSuccessors(current_state[0]):
            if grid[successor[0]] == 0:
                l = list(successor)
                l.append(current_state[2])
                element = (successor[2], l)
                state_queue.put(element)
                grid[successor[0]] = 1

    final_actions = []

    current_action = actions[-1]

    while current_action[2] != 0:
        final_actions.insert(0, current_action[1])
        current_action = actions[current_action[-1]]

    return final_actions

import numpy as np
def nullHeuristic(state, problem=None):

    return np.abs(state[0] - problem.goal[0]) + np.abs(state[1] - problem.goal[1])


def aStarSearch(problem, heuristic=nullHeuristic):

    actions = []
    state_queue = PriorityQueue()
    state_queue.put((heuristic(problem.getStartState(), problem), [problem.getStartState(), "Stop", 0, 0]))
    grid = np.zeros((1000, 1000))

    while not state_queue.empty():

        current_state = state_queue.get()[1]
        current_state[2] = len(actions)
        actions.append(current_state)

        if problem.isGoalState(current_state[0]):
            break

        for successor in problem.getSuccessors(current_state[0]):
            if grid[successor[0]] == 0:
                l = list(successor)
                l.append(current_state[2])
                element = (successor[2] + heuristic(successor[0], problem), l)
                state_queue.put(element)
                grid[successor[0]] = 1

    final_actions = []

    current_action = actions[-1]

    while current_action[2] != 0:
        final_actions.insert(0, current_action[1])
        current_action = actions[current_action[-1]]

    return final_actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
