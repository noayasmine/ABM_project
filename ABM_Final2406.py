#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Francijn

24/06 - in each time step; an agent has an encounter with someone.
The choice with whom an agent has an encounter with is probabilistic:
and based on popularity with weight p, and path length, with weight 1 - p.

Then depending on whether this encountered agent is already being followed or not;
the agent takes an action:
    if not following yet:
        if agents are similar (within the threshold decided per simulation):
            follow
        else
            do nothing
    if already following:
        if engagement is lower than engagement threshold (also set per simulation):
            unfollow
        else
            do nothing

By changing the threshold values and the weights of the importance of popularity and pathlength in having an encounter
we can see what values lead to network structures with highly popular individuals (influencers)
NOT SURE YET WHAT EXACT RESEARCH QUESTION WOULD MAKE SENSE THOUGH
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as r
from collections import defaultdict

class SocialNetwork():
    def __init__(self, n_agents, prob, concentration, follow_threshold, unfollow_threshold, w_pop, w_prox):
        self.n_agents = n_agents
        self.prob = prob
        self.concentration = concentration
        self.follow_threshold = follow_threshold
        self.unfollow_threshold = unfollow_threshold
        self.w_pop = w_pop
        self.w_prox = w_prox

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


    def lowest_utility_candidates(self, unique_id):
        candidates = self.SHORTEST_PATH[unique_id]
        # get neighbors
        candidates = {key: value for key, value in candidates.items() if value in {1}}
        engagement_scores = {candidate: self.WEIGHT[unique_id][candidate] for candidate in candidates}
        random_candidate = r.choice(list(candidates.keys()))
        
        return engagement_scores[random_candidate], random_candidate
    
    def decide_to_follow(self, follower, followee, follow_threshold):
        if abs(self.OPINIONS[follower] - self.OPINIONS[followee]) <= follow_threshold:
            return True
        else:
            return False

    def decide_to_unfollow(self, follower, followee, unfollow_threshold):
        if self.WEIGHT[follower][followee] <= unfollow_threshold:
            return True
        else:
            return False

    def have_encounter_with(self, agent, w_pop, w_prox):
        probs = []
        agents = []
        for followee in self.G.nodes():
            # chances of an agent following themselves is 0
            agents.append(followee)
            if agent == followee:
                probs.append(0)
            else:
            # popularity is based on the number of followers / total who could follow you (so n_agents)
                U_popularity = self.IN[followee] / self.n_agents
        #U_similarity = 1 - abs(self.OPINIONS[follower] - self.OPINIONS[followee])
        #U_engagement = sum(self.WEIGHT[follower][n] * self.WEIGHT[followee][n] for n in set(self.WEIGHT[follower]) & set(self.WEIGHT[followee]))
                # if shortest path is 1 or 2, then they have an equal probablity to be encountered
                # if shortest path is longer than that, than probability of encounter is relative to path length
                if self.SHORTEST_PATH[agent][followee] == 1:
                    U_proximity = 1 - (2/100)
                else:
                    U_proximity = 1  - (self.SHORTEST_PATH[agent][followee]/100)

                U_total = (w_pop * U_popularity + 
                   #w_sim * U_similarity)
                   #w_engagement * U_engagement + 
                   w_prox * U_proximity)
            
                probs.append(U_total)
        
        norm_probs = []
        # normalize probabilities
        for prob in probs:
            norm_probs.append((prob/sum(probs)))

        encounter = r.choices(agents, weights=norm_probs, k=1)
        return encounter[0]
    
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
            # if you are already at the max of your following number; 
            # unfollow someone (with whom you have low engagement)
            if self.G.out_degree(node) >= self.MAX[node]:
                lowest_utility_candidate = self.lowest_utility_candidates(node)
                if lowest_utility_candidate is not None:
                    edges_to_remove.append((node, lowest_utility_candidate))
            else:
                # choose an agent based (higher probability to be picked depending on
                # path length and popularity)
                encountered = self.have_encounter_with(node, self.w_pop, self.w_prox)
                # in this case the agent does not follow the potential followee yet
                if self.SHORTEST_PATH[node][encountered] > 1:
                # if the agent does not follow the person yet; 
                # decide if you want to follow (based on opinion similarity)
                        if self.decide_to_follow(node, encountered, self.follow_threshold):
                            edges_to_add.append((node, encountered))
                # if the agent does already follow, check if you want to keep following
                # based on engagement
                elif self.SHORTEST_PATH[node][encountered] == 1:
                    if self.decide_to_unfollow(node, encountered, self.unfollow_threshold):
                        edges_to_remove.append((node, encountered))
                    
        # for computational reasons first all agents do an action
        # after that the actual network is changed
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

# not sure what this is for exactly
concentration = 2.5

# how close an opinion has to be to another agents opinion to decide to follow them
follow_threshold = 0.2

# if engagement is lower than this threshold, the decision is made to unfollow
unfollow_threshold = 0.1

# how important either of these things is (this decides whom to have an interaction with)
w_popularity = 0.5
w_proximity = 1 - w_popularity

# Initialize and run the model
model = SocialNetwork(n_agents, prob, concentration, follow_threshold, unfollow_threshold, w_popularity, w_proximity)

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