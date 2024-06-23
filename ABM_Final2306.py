#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:58:09 2024

@author: noaroebersen
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as r
from collections import defaultdict

class SocialNetwork():
    def __init__(self, n_agents, prob, concentration=2.5):
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
        self.OPINIONS = {i: r.uniform(0, 1) for i in range(n_agents)}
        self.MAX = defaultdict(int)
        
        self.SHORTEST_PATH = defaultdict(int)
        self.Data_Collector = {"avg utility": [], "avg IN degrees": [], "avg OUT degrees" : [], 
                               "avg clustering coeff" : []}
        self.create_random_network()
    
    def create_random_network(self):
        self.G = nx.gnp_random_graph(n=self.n_agents, p=self.prob, directed=True)
        # add weights to the edges
        for node in self.G.nodes():
            out_edges = list(self.G.out_edges(node))
            if out_edges:
                engagements = np.random.dirichlet(np.full(len(out_edges), self.concentration))
                for (u, v), engagement in zip(out_edges, engagements):
                    self.WEIGHT[u][v] = engagement
                    self.UTILITIES[v][u] = engagement
                    self.OUT[u] += 1
                    self.IN[v] += 1
            self.MAX[node] = r.choice(range(self.G.out_degree(node), n_agents))
        self.SHORTEST_PATH = dict(nx.shortest_path_length(self.G))
    
    
    def normalize_weights(self):
        for follower in self.WEIGHT:
            total_engagement = sum(self.WEIGHT[follower].values())
            self.WEIGHT[follower] = {influencer : engagement / total_engagement for 
                                          influencer, engagement in self.WEIGHT[follower].items()}
        for follower in self.WEIGHT:
            for influencer in self.WEIGHT[follower]:
                self.UTILITIES[influencer][follower] = self.WEIGHT[follower][influencer]

        
    
    def utility_score(self, follower, followee):
        w_popularity = 0.5
        w_similarity = 0.5
        #w_engagement = 0.2
        #w_proximity = 0.2
        
        # normalize popularity by in_connections / total_connections
        U_popularity = self.IN[followee] / (self.IN[followee] + self.OUT[followee])
        U_similarity = 1 - abs(self.OPINIONS[follower] - self.OPINIONS[followee])
        #U_engagement = sum(self.WEIGHT[follower][n] * self.WEIGHT[followee][n] for n in set(self.WEIGHT[follower]) & set(self.WEIGHT[followee]))
        #U_proximity = 1 / (self.SHORTEST_PATH[follower][followee] + 1)

        U_total = (w_popularity * U_popularity + 
                   w_similarity * U_similarity)
                   #w_engagement * U_engagement + 
                   #w_proximity * U_proximity)
        
        return U_total
    
    def highest_utility_candidates(self, unique_id):
        candidates = self.SHORTEST_PATH[unique_id]
        # now modelled as a 'seeing' of 3 i.e. take into account 'neighbors of neighbors' and and 'friends of friends of friends'

        candidates = {key: value for key, value in candidates.items() if value in {2,3}}

        if not candidates:
            return None

        utility_scores = {candidate: self.utility_score(unique_id, candidate) for candidate in candidates}
        max_utility_candidate = max(utility_scores, key=utility_scores.get)
        #print(utility_scores[max_utility_candidate], max_utility_candidate)
        return utility_scores[max_utility_candidate], max_utility_candidate
    
    def highest_utility_candidates_v2(self, unique_id):
        candidates = self.SHORTEST_PATH[unique_id]
        # now modelled as a 'seeing' of 3 i.e. take into account 'neighbors of neighbors' and and 'friends of friends of friends'

        candidates = {key: value for key, value in candidates.items() if value in {2,3}}
        
        if not candidates:
            return None

        random_candidate = r.choice(list(candidates.keys()))

        utility_score = self.utility_score(unique_id, random_candidate)

        return utility_score, random_candidate
    

        
    def lowest_utility_candidates(self, unique_id):
        candidates = self.SHORTEST_PATH[unique_id]
        # get neighbors
        candidates = {key: value for key, value in candidates.items() if value in {1}}
        engagement_scores = {candidate: self.WEIGHT[unique_id][candidate] for candidate in candidates}
        min_engagement_candidate = max(engagement_scores, key=engagement_scores.get)
        
        
        return engagement_scores[min_engagement_candidate], min_engagement_candidate
    
    def lowest_utility_candidates_v2(self, unique_id):
        candidates = self.SHORTEST_PATH[unique_id]
        # get neighbors
        candidates = {key: value for key, value in candidates.items() if value in {1}}
        engagement_scores = {candidate: self.WEIGHT[unique_id][candidate] for candidate in candidates}
        random_candidate = r.choice(list(candidates.keys()))
        
        return engagement_scores[random_candidate], random_candidate



    def add_connection(self, follower, followee):
        if not self.G.has_edge(follower, followee):
            self.G.add_edge(follower, followee)
            engagement = r.uniform(0, 1)
            self.WEIGHT[follower][followee] = engagement
            self.UTILITIES[followee][follower] = engagement
            self.OUT[follower] += 1
            self.IN[followee] += 1

    def remove_connection(self, follower, followee):
        if self.G.has_edge(follower, followee):
            self.G.remove_edge(follower, followee)
            del self.WEIGHT[follower][followee]
            del self.UTILITIES[followee][follower]
            self.OUT[follower] -= 1
            self.IN[followee] -= 1

    def track_metrics(self):
        avg_in_degree = sum(self.IN.values()) / self.n_agents
        avg_out_degree = sum(self.OUT.values()) / self.n_agents
        avg_clustering_coeff = nx.average_clustering(self.G)
        avg_utility = sum(sum(inner_dict.values()) for inner_dict in self.UTILITIES.values()) / self.n_agents

        self.Data_Collector["avg IN degrees"].append(avg_in_degree)
        self.Data_Collector["avg OUT degrees"].append(avg_out_degree)
        self.Data_Collector["avg clustering coeff"].append(avg_clustering_coeff)
        self.Data_Collector["avg utility"].append(avg_utility)

    def step(self):
        edges_to_add = []
        edges_to_remove = []

        for node in self.G.nodes():
            # use logit to map difference of in and out degree to [0,1]
            # do we want to use this?
            utility = 1 / (1 + np.exp(-(self.IN[node] - self.OUT[node])))

            if self.G.out_degree(node) >= self.MAX[node]:
                lowest_utility_candidate = self.lowest_utility_candidates_v2(node)
                if lowest_utility_candidate is not None:
                    edges_to_remove.append((node, lowest_utility_candidate))
            else:
                #highest_utility, highest_utility_candidate = self.highest_utility_candidates(node)
                highest_utility_result = self.highest_utility_candidates_v2(node)
                if highest_utility_result is not None:
                    highest_utility, highest_utility_candidate = highest_utility_result
                    if highest_utility > utility:
                        #print('follow')
                        edges_to_add.append((node, highest_utility_candidate))
                        
                # how do we define this?
                #lowest_utility, lowest_utility_candidate = self.lowest_utility_candidates(node)
                lowest_utility_result = self.lowest_utility_candidates_v2(node)
                if lowest_utility_result is not None:
                    lowest_utility, lowest_utility_candidate = lowest_utility_result
                    if lowest_utility < utility:
                        #print('unfollow')
                        edges_to_remove.append((node, lowest_utility_candidate))
                    

        for follower, followee in edges_to_add:
            self.add_connection(follower, followee)

        for follower, exfollowee in edges_to_remove:
            self.remove_connection(follower, exfollowee)

        self.normalize_weights()
        self.track_metrics()
        self.SHORTEST_PATH = dict(nx.shortest_path_length(self.G))

# Parameters
steps = 200
n_agents = 100
avg_degree = 25
prob = avg_degree / n_agents

# Initialize and run the model
model = SocialNetwork(n_agents, prob)

for i in range(steps + 1):
    model.step()
    print(f"\rProgress: {(i / steps) * 100:.2f}%", end='', flush=True)

# Save results to CSV
df_results = pd.DataFrame(model.Data_Collector)
df_results.to_csv("social_network_metrics.csv", index=False)

# Plotting metrics
fig, axs = plt.subplots(4, 1, figsize=(5, 20))

axs[0].plot(df_results.index, df_results["avg OUT degrees"], label='Average Out-Degree')
#axs[0].set_title('Average Out-Degree over Time')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Average Out-Degree')
axs[0].legend()

axs[1].plot(df_results.index, df_results["avg IN degrees"], label='Average In-Degree', color='orange')
#axs[1].set_title('Average In-Degree over Time')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Average In-Degree')
axs[1].legend()

axs[2].plot(df_results.index, df_results["avg clustering coeff"], label='Average Clustering Coefficient', color='green')
#axs[2].set_title('Average Clustering Coefficient over Time')
axs[2].set_xlabel('Time Step')
axs[2].set_ylabel('Clustering Coefficient')
axs[2].legend()

axs[3].plot(df_results.index, df_results["avg utility"], label='Average Utility', color='blue')
#axs[3].set_title('Average Utility over Time')
axs[3].set_xlabel('Time Step')
axs[3].set_ylabel('Utility')
axs[3].legend()

#plt.subplots_adjust(hspace=5.0)
plt.tight_layout()
plt.show()

# Plot the distribution of utility values
all_utilities = [utility for influencer in model.UTILITIES.values() for utility in influencer.values()]

plt.figure(figsize=(10, 6))
sns.histplot(all_utilities, kde=True, bins=30)
plt.title('Distribution of Utility Values')
plt.xlabel('Utility')
plt.ylabel('Frequency')
plt.show()






# Plot the network itself
def plot_network(G, WEIGHT):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)  # positions for all nodes

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", 
            node_shape="o", alpha=0.75, linewidths=4)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Create edge labels using weights from the WEIGHT attribute
    edge_labels = {(u, v): f"{WEIGHT[u][v]:.2f}" for u, v in G.edges()}
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Social Network Graph")
    plt.show()

# Check if the network is fully connected
def check_network_connectivity(G):
    if nx.is_strongly_connected(G):
        print("The network is strongly connected: there is a path from every node to every other node.")
    else:
        print("The network is not strongly connected.")

# Plot the distribution of connections per agent
def plot_degree_distribution(G):
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(in_degrees, bins=100, kde=False)
    plt.title('In-Degree Distribution')
    plt.xlabel('In-Degree')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.histplot(out_degrees, bins=100, kde=False)
    plt.title('Out-Degree Distribution')
    plt.xlabel('Out-Degree')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Use the functions
# Use the function with the modified edge labels
#plot_network(model.G, model.WEIGHT)
#check_network_connectivity(model.G)
plot_degree_distribution(model.G)