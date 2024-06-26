from ABM_Final2206 import SocialNetwork
import networkx as nx
import matplotlib.pyplot as plt

steps = 10
n_agents = 12
avg_degree = 3
prob = avg_degree/n_agents

model = SocialNetwork(n_agents, prob)

pos = nx.spring_layout(model.G)

for i in range(steps):
    plt.clf()
    model.step()
    nx.draw(model.G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=15)
    plt.title(f"Iteration {i+1}")
    plt.pause(1.5)

plt.show()