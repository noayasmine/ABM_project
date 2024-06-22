import networkx as nx
import matplotlib.pyplot as plt
import random as r
import numpy as np
import pandas as pd
import seaborn as sns
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

        self.SHORTHEST_PATH = defaultdict(int)

        # self.Data_Collector = {"avg utility": [], "avg IN degrees": [], "avg OUT degrees" : [], 
        #                        "avg path length" : [], "avg clustering coeff" : []}
        self.Data_Collector = {"avg utility": [], "avg IN degrees": [], "avg OUT degrees" : [], 
                               "avg clustering coeff" : []}

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
        self.SHORTHEST_PATH = dict(nx.shortest_path_length(self.G))
    
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

    def choose_to_follow(self, unique_id):

        # this is now only based on how far away someone is
        # if path is shorter: more likely to make connection
        options = self.SHORTHEST_PATH[unique_id]

        # remove the nodes that are already directly connected
        filtered_options = {key: value for key, value in options.items() if value not in {0, 1}}

        # check if the node is not already connected to every other node
        if len(filtered_options) != 0:
            nodes_shortest_path = list(filtered_options.keys())
            distances_shortest_path_id = list(filtered_options.values())
            probs = [1 - (dist / 100) for dist in distances_shortest_path_id]
            total_prob = sum(probs)
            probs_norm = [prob / total_prob for prob in probs]

            followee = np.random.choice(nodes_shortest_path, p=probs_norm)
            return followee
        else:
            return (unique_id + 1)

    def choose_to_unfollow(self, unique_id):
        """for now it chooses someone completely random to unfollow"""
        list = self.G.edges(unique_id)
        options = []
        for i in list:
            options.append(i[1])
        if len(options) != 0:
            return r.choice(options)
        else:
            return

    def add_connection(self, follower, followee):
        if not self.G.has_edge(follower, followee):
            self.G.add_edge(follower, followee)
            engagement = r.uniform(0, 1)
            self.OUT[follower] += 1
            self.WEIGHT[follower][followee] = engagement
            self.IN[followee] += 1
            self.UTILITIES[followee][follower] = engagement

    def remove_connection(self, follower, exfollowee):
        if self.G.has_edge(follower, exfollowee):
            self.G.remove_edge(follower, exfollowee)
            self.OUT[follower] -= 1
            del self.WEIGHT[follower][exfollowee]
            self.IN[exfollowee] -= 1
            del self.UTILITIES[exfollowee][follower]
    
    def track_metrics(self):
        # add to data collector
        self.Data_Collector["avg IN degrees"].append(sum(self.IN.values()) / self.n_agents)
        self.Data_Collector["avg OUT degrees"].append(sum(self.OUT.values()) / self.n_agents)
        #self.Data_Collector["avg path length"].append(nx.average_shortest_path_length(self.G))
        self.Data_Collector["avg clustering coeff"].append(nx.average_clustering(self.G))
        self.Data_Collector["avg utility"].append(sum(sum(inner_dict.values()) 
                                                         for inner_dict in self.UTILITIES.values()) / self.n_agents)
        
    def step(self):
        edges_to_add = []
        edges_to_remove = []
        # each agent does an action
        for node in self.G.nodes():
            in_engagement = sum(self.UTILITIES[node].values())
            # this strategy needs to become something more realistic and less simple
            # 
            if self.IN[node] > self.OUT[node]:
                followee = self.choose_to_follow(node)
                edges_to_add.append((node, followee))
            elif self.IN[node] < self.OUT[node]:
                exfollowee = self.choose_to_unfollow(node)
                edges_to_remove.append((node, exfollowee))
        
        for i in edges_to_add:
            self.add_connection(i[0], i[1])

        for i in edges_to_remove:
            self.remove_connection(i[0], i[1])
            # calculate which strategy to use
            # implement strategy
        self.normalize_weights()
        self.track_metrics()
        # update shortest path
        self.SHORTHEST_PATH = dict(nx.shortest_path_length(self.G))

    # def evaluate_received_engagement(self, agent):
        
steps = 200
n_agents = 100
avg_degree = 25
prob = avg_degree/n_agents

model = SocialNetwork(n_agents, prob)
for i in range(steps+1):
    model.step()
    print(f"\r{(i/steps)*100:.2f}%", end='', flush=True)

df_results = pd.DataFrame(model.Data_Collector)
df_results.to_csv("test.csv")
#print(df_results)

#pos = nx.spring_layout(model.G)
#nx.draw(model.G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=15)
#plt.show()

ig, axs = plt.subplots(4, 1, figsize=(10, 50))

axs[0].plot(df_results.index, df_results["avg OUT degrees"], label='Average Out-Degree')
axs[0].set_title('Average Out-Degree over Time')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Average Out-Degree')
axs[0].legend()

axs[1].plot(df_results.index, df_results["avg IN degrees"], label='Average In-Degree', color='orange')
axs[1].set_title('Average In-Degree over Time')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Average In-Degree')
axs[1].legend()

axs[2].plot(df_results.index, df_results["avg clustering coeff"], label='Average Clustering Coefficient', color='green')
axs[2].set_title('Average Clustering Coefficient over Time')
axs[2].set_xlabel('Time Step')
axs[2].set_ylabel('Clustering Coefficient')
axs[2].legend()

axs[3].plot(df_results.index, df_results["avg utility"], label='Utility', color='blue')
axs[3].set_title('Average Utility over Time')
axs[3].set_xlabel('Time Step')
axs[3].set_ylabel('Utility')
axs[3].legend()

#axs[4].plot(df_results.index, df_results["avg path length"], label='Average Path Length', color='red')
#axs[4].set_title('Average Path Length over Time')
#axs[4].set_xlabel('Time Step')
#axs[4].set_ylabel('Path Length')
#axs[4].legend()

plt.tight_layout()
plt.show()


