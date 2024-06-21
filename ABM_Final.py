import networkx as nx
import matplotlib.pyplot as plt
import random as r
import numpy as np

# def create_random_network(n_agents, prob, concentration):
#     G = nx.gnp_random_graph(n=n_agents, p=prob, directed=True)
#     # add weights to the edges
#     for node in G.nodes():
#         out_edges = list(G.out_edges(node))
#         if out_edges:
#             # concentration affects the distribution of engagement (0.5 = uneven, 10.0 = even)
#             engagements = np.random.dirichlet(np.full(len(out_edges), concentration))
#             for (u, v), engagement in zip(out_edges, engagements):
#                 G.edges[u, v]['weight'] = engagement
#     return G

# def normalize_weights(G: nx.DiGraph) -> nx.DiGraph:
#     for node in G.nodes:
#         out_edges = list(G.out_edges(node, data=True))
#         total_weight = sum(edge[2]['weight'] for edge in out_edges)
#         for edge in out_edges:
#             edge[2]['weight'] = round(edge[2]['weight'] / total_weight, 3)
#     return G

class SocialNetwork():
    def __init__(self, n_agents, prob, concentration = 2.5):
        self.n_agents = n_agents
        self.prob = prob
        self.concentration = concentration

        self.create_random_network()
    
    def create_random_network(self):
        self.G = nx.gnp_random_graph(n=self.n_agents, p=self.prob, directed=True)
        # add weights to the edges
        for node in self.G.nodes():
            out_edges = list(self.G.out_edges(node))
            if out_edges:
                # concentration affects the distribution of engagement (0.5 = uneven, 10.0 = even)
                engagements = np.random.dirichlet(np.full(len(out_edges), self.concentration))
                for (u, v), engagement in zip(out_edges, engagements):
                    self.G.edges[u, v]['weight'] = engagement
    
    def normalize_weights(self):
        for node in self.G.nodes:
            out_edges = list(self.G.out_edges(node, data=True))
            total_weight = sum(edge[2]['weight'] for edge in out_edges)
            for edge in out_edges:
                edge[2]['weight'] = round(edge[2]['weight'] / total_weight, 3)


"""
{NODE1: {Followed_NODE1 : engagement}, Followed_NODE2 : engagement, 
    NODE2: {Followed_NODE1 : engagement}, Followed_NODE2 : engagement}
}
"""

n_agents = 10
avg_degree = 5
prob = avg_degree/n_agents
G = SocialNetwork(n_agents, prob)
G.normalize_weights()

pos = nx.spring_layout(G.G)
nx.draw(G.G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=15)
plt.show()