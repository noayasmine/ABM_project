import networkx as nx
import matplotlib.pyplot as plt
import random as r
import numpy as np
from collections import defaultdict

class SocialNetwork():
    def __init__(self, n_agents, prob, concentration = 2.5):
        self.n_agents = n_agents
        self.prob = prob
        self.concentration = concentration

        # {Node1 : IN_Degree, Node2 : IN_Degree}
        self.IN = defaultdict(int)
        # {Node1 : OUT_Degree, Node2 : OUT_Degree}
        self.OUT = defaultdict(int)
        # {Node1 : {Follower1 : Engagement, Follower2 : Engagement}, Node2 : ETC}
        # for user engagement: self.WEIGHT[follower][influencer]
        self.WEIGHT = defaultdict(lambda: defaultdict(float))
        # for engagement received from follower: self.UTILITIES[influencer][follower]
        # for total engagement: sum(self.UTILITIES[influencer].values())
        self.UTILITIES = defaultdict(lambda: defaultdict(float))

        self.Data_Collector = {"avg utility": [], "avg IN degrees": [], "avg OUT degrees" : [], 
                               "avg path length" : [], "avg clustering coeff" : []}

        self.create_random_network()
    
    def create_random_network(self):
        self.G = nx.gnp_random_graph(n=self.n_agents, p=self.prob, directed=True)
        # add weights to the edges
        for node in self.G.nodes():
            out_edges = list(self.G.out_edges(node))
            # self.OUT[node] = len(out_edges)
            if out_edges:
                # concentration affects the distribution of engagement (0.5 = uneven, 10.0 = even)
                engagements = np.random.dirichlet(np.full(len(out_edges), self.concentration))
                for (u, v), engagement in zip(out_edges, engagements):
                    self.WEIGHT[u][v] = engagement
                    self.UTILITIES[v][u] = engagement
                    self.OUT[u] += 1
                    self.IN[v] += 1
    
    # to be done after each time step
    def normalize_weights(self):
        # normalize weights for each agent
        for follower in self.WEIGHT:
            total_engagement = sum(self.WEIGHT[follower].values())
            self.WEIGHT[follower] = {influencer : engagement / total_engagement for 
                                          influencer, engagement in self.WEIGHT[follower].items()}
        # update received utilities
        for follower in self.WEIGHT:
            for influencer in self.WEIGHT[follower]:
                self.UTILITIES[influencer][follower] = self.WEIGHT[follower][influencer]
                
    def add_connection(self, follower, followee):
        if not self.G.has_edge(follower, followee):
            self.G.add_edge(follower, followee)
            engagement = np.random.dirichlet(np.full(1, self.concentration))
            self.OUT[follower] += 1
            self.WEIGHT[follower][followee] = engagement
            self.IN[followee] += 1
            self.UTILITIES[followee][follower] = engagement

    def remove_connection(self, follower, exfollowee):
        if not self.G.has_edge(follower, exfollowee):
            self.G.remove_edge(follower, exfollowee)
            self.OUT[follower] -= 1
            del self.WEIGHT[follower][exfollowee]
            self.IN[exfollowee] -= 1
            del self.UTILITIES[exfollowee][follower]
    
    def track_metrics(self):
        # add to data collector
        self.Data_Collector["avg IN degrees"].append(sum(self.IN.values()) / self.n_agents)
        self.Data_Collector["avg OUT degrees"].append(sum(self.OUT.values()) / self.n_agents)
        self.Data_Collector["avg path length"].append(nx.average_shortest_path_length(self.G))
        self.Data_Collector["avg clustering coeff"].append(nx.average_clustering(self.G))
        self.Data_Collector["avg utility"].append(sum(sum(inner_dict.values()) 
                                                         for inner_dict in self.UTILITIES.values()) / self.n_agents)
        
    def step(self):
        for node in self.G.nodes():
            # calculate which strategy to use
            # implement strategy
            pass
        self.normalize_weights()
        self.track_metrics()
        # print(self.Data_Collector["avg IN degrees"])

    # def evaluate_received_engagement(self, agent):
        
steps = 2
n_agents = 200
avg_degree = 15
prob = avg_degree/n_agents

model = SocialNetwork(n_agents, prob)
for i in range(steps):
    model.step()
    print(f"\r{(i/steps)*100:.2f}%", end='', flush=True)

# pos = nx.spring_layout(model.G)
# nx.draw(model.G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=15)
# plt.show()