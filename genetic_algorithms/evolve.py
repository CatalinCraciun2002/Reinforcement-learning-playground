import numpy as np

def softmax(scores):

    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def select_agents(agents, scores, percentage=0.4, nr_agents_of_agents=None):

    if nr_agents_of_agents is not None:
        num_agents_to_select = int(len(agents) * percentage)
    else:
        num_agents_to_select = nr_agents_of_agents

    selected_agents_with_scores = sorted(zip(agents, scores), reverse=True, key=lambda x: x[1])[:num_agents_to_select]

    return selected_agents_with_scores


import random
import copy
import random

def crossover_and_mutate(agents, random_factor):
    child_agent = copy.deepcopy(agents[0])

    distribution = softmax(np.random.randn(len(agents)))

    random_value = random.uniform(0, random_factor)

    distribution*= (1 - random_value)

    for i, agent in enumerate(agents):

        for params_child, params_agent  in zip(child_agent.model.parameters(), agent.model.parameters()):
            for j in range(params_child.shape[0]):

                if i == 0:
                    params_child.data[j] *= distribution[i]
                else:
                    params_child.data[j] += params_agent.data[j] * distribution[i]


    for params_child in child_agent.model.parameters():
        for j in range(params_child.shape[0]):
            shape = tuple(params_child.data[j].shape)
            params_child.data[j] += (2 * np.random.rand(*shape) -1) * random_value

    return child_agent

from tqdm import tqdm
def generate_new_population(selected_agents_with_sores, num_offspring, random_factor):

    selected_agents= [agent for agent, score in selected_agents_with_sores]
    selected_scores = [score for agent, score in selected_agents_with_sores]

    selected_scores = np.array(selected_scores)
    selected_scores = selected_scores / np.max(np.abs(selected_scores)) * 10
    selected_scores = softmax(selected_scores)

    new_population = []
    for i in tqdm(range(num_offspring)):

        selected_agents_run = np.random.choice(len(selected_agents), 2, p=selected_scores)
        selected_agents_run = [selected_agents[i] for i in selected_agents_run]

        child_agent = crossover_and_mutate(selected_agents_run, random_factor)
        new_population.append(child_agent)

    return new_population


def genetic_algorithm_evolution(agents, scores, agents_to_generate, random_factor=0.1, percentage=0.4):
    selected_agents_with_sores = select_agents(agents, scores, percentage)
    new_agents = generate_new_population(selected_agents_with_sores, agents_to_generate, random_factor)
    return new_agents


