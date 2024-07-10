import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import networkx as nx
from collections import defaultdict
import random as r

k_B = 8.617333262145e-5  # Boltzmann constant in eV/K

n_agents = 75
avg_degree = 3
prob = avg_degree / n_agents

IN = defaultdict(int)
        # {Node1 : OUT_Degree, Node2 : OUT_Degree}
OUT = defaultdict(int)
        # {Node1 : {Follower1 : Engagement, Follower2 : Engagement}, Node2 : ETC}
        # for user engagement: self.WEIGHT[follower][influencer]
WEIGHT = defaultdict(lambda: defaultdict(float))
        # for engagement received from follower: self.UTILITIES[influencer][follower]
        # for total engagement: sum(self.UTILITIES[influencer].values())
UTILITIES = defaultdict(lambda: defaultdict(float))
OPINIONS = {i: r.uniform(0, 1) for i in range(n_agents)}
MAX = defaultdict(int)
SHORTEST_PATH = defaultdict(int)


G = nx.gnp_random_graph(n=n_agents, p=prob, directed=True)
        # add weights to the edges
for node in G.nodes():
    out_edges = list(G.out_edges(node))
    if out_edges:
        engagements = np.random.uniform(0,1,len(out_edges))
        for (u, v), engagement in zip(out_edges, engagements):
            WEIGHT[u][v] = engagement
            UTILITIES[v][u] = engagement
            OUT[u] += 1
            IN[v] += 1
    MAX[node] = r.choice(range(G.out_degree(node), n_agents))
    SHORTEST_PATH = dict(nx.shortest_path_length(G))

# getting the value of the longest shortest path

# Initialize a variable to store the maximum value
max_value = float('-inf')

# Iterate through each inner dictionary
for inner_dict in SHORTEST_PATH.values():
    # Get the maximum value in the current inner dictionary and compare it with max_value
    max_value = max(max_value, max(inner_dict.values()))

print("The maximum value in all inner dictionaries is:", max_value)

agent = 1
agents = list(G.nodes())
agents.remove(agent)  # Removing the agent itself from the list

distances = []
for followee in G.nodes():
    if followee == agent:
        continue
    else:
        if followee in SHORTEST_PATH[agent]:
            distances.append(SHORTEST_PATH[agent][followee])
        else:
                # have number of weights match the number of agents 
            agents.remove(followee)

    
distances = list(set(distances))
distances = np.array(distances)
print(distances)


def transform_distances(distances, mu, temperature, max_distance):
    exponents = (distances / max_value - mu) / temperature
    probs = expit(-exponents)
    return probs

# Example distances
#distances = np.linspace(0, 100, 500)

# Define different values of mu and temperature
mu_values = [0.4]  # Chemical potential in the normalized range
temperature_values = [0.1]  # Temperature in arbitrary units

# Maximum distance for normalization
if len(distances) != 0:
    max_distance = np.max(distances)

# Plot the transformed distances for different mu and temperature
    plt.figure(figsize=(10, 6))

    for mu in mu_values:
        for temperature in temperature_values:
            probs = transform_distances(distances, mu, temperature, max_distance)
            plt.plot(distances, probs, 'o-', label=f'$\mu$ = {mu}, T = {temperature}')

    plt.xlabel('Distances')
    plt.ylabel('Transformed Probabilities')
    plt.title('Transformed Distances using Sigmoid Function')
    plt.legend()
    plt.grid(True)
    plt.show()
