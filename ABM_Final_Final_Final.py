#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Francijn


24/06 - in each time step; an agent has an encounter with someone.
The choice with whom an agent has an encounter with is probabilistic:
and based on path length --> fermi durac distribution


Then depending on whether this encountered agent is already being followed or not;
the agent takes an action:
    if not following yet:
        if total_utility > random_uniform_utility:
            follow
        else
            do nothing
    if already following:
        if engagement is lower than engagement threshold (also set per simulation):
            unfollow
        else
            do nothing


"""

import numpy as np
import pandas as pd
import networkx as nx
import random as r
from collections import defaultdict
from scipy.special import expit


class SocialNetwork():
    def __init__(self, n_agents, prob, w_pop, w_prox, w_sim, mu, temp, sociability):
        self.n_agents = n_agents
        self.prob = prob
        self.w_pop = w_pop
        self.w_prox = w_prox
        self.w_sim = w_sim
        self.prob = prob
        self.mu = mu
        self.temperature = temp

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

        #self.CONFORMITY = {i: np.random.exponential(1/3)/10+conf_mu for i in range(n_agents)}
        #self.TOLERANCE = {i: np.random.exponential(1/3)/10+tol_mu for i in range(n_agents)}
        self.SOCIABILITY = sociability
        #self.TOLERANCE = sociability
        #self.decay_factor = {i: np.random.uniform(0.9,1) for i in range(n_agents)}
        #self.activity = {i: np.random.uniform(0,1) for i in range(n_agents)}
        # print(self.CONFORMITY)
        # print(self.TOLERANCE)

        self.SHORTEST_PATH = defaultdict(int)
        self.Data_Collector = {"max IN degrees": [], "avg degrees": [],
                               "avg clustering coeff" : [], "betweenness centrality" : [], 
                               "degree sequence": [], "IN degree": []}
        self.create_random_network()
   
    def create_random_network(self):
        self.G = nx.gnp_random_graph(n=self.n_agents, p=self.prob, directed=True)
        #self.G = nx.scale_free_graph(n=self.n_agents)
        # add weights to the edges
        for node in self.G.nodes():
            out_edges = list(self.G.out_edges(node))
            if out_edges:
                engagements = np.random.uniform(0,1,len(out_edges))
                for (u, v), engagement in zip(out_edges, engagements):
                    self.WEIGHT[u][v] = engagement
                    self.UTILITIES[v][u] = engagement
                    self.OUT[u] += 1
                    self.IN[v] += 1
            self.MAX[node] = r.choice(range(self.G.out_degree(node), self.n_agents))
        self.SHORTEST_PATH = dict(nx.shortest_path_length(self.G))
   
    def update_opinions_and_weights(self):
        # conformity is the rate an agent changes their opinion to match their neighborhood
        # we probably should make it a random parameter for each agent
        # or a global parameter?
        # conformity = 0.1
        for j, node in enumerate(self.G.nodes()):
            conformity = self.SOCIABILITY
            tolerance = self.SOCIABILITY
            opinions = []
            weights = []
            
            for i in self.WEIGHT[node]:
                if abs(self.OPINIONS[i] - self.OPINIONS[node]) > tolerance:
                    continue
                opinions.append(self.OPINIONS[i])
                weights.append(self.WEIGHT[node][i])
                
            if sum(weights) == 0:
                consensus = 0.5
            else:
                consensus = np.average(opinions, weights=weights)
            old_opinion = self.OPINIONS[node]
            new_opinion = old_opinion + conformity*(consensus - old_opinion) + np.random.normal(0,0.01)
            self.OPINIONS[node] = np.clip(new_opinion, 0, 1)
       
            # update weights based on difference of opinion
            for i in self.WEIGHT[node]:
                new_weight = self.WEIGHT[node][i]
                # weights between 0 and 1
                min_weight = np.min(weights) if len(weights) != 0 else 0
                max_weight = np.max(weights) if len(weights) != 0 else 1
                if max_weight == 0:
                    max_weight = 1
                    min_weight = 0
                # otherwise it threw some errors when the average degree was low by initializing
                if max_weight - min_weight == 0:
                    continue
                else:
                    self.WEIGHT[node][i] = (new_weight-min_weight+0.001)/(max_weight-min_weight+0.001)
                #print((0.95 + (self.IN[i] / max(self.IN) * 0.05)))
                
                self.WEIGHT[node][i] *= ((1-self.SOCIABILITY) + (self.IN[i] / max(self.IN) * self.SOCIABILITY))

    def utility_score(self, follower, followee):
        # normalize popularity by in_connections / total_connections
     
        #U_popularity = 0.5 * self.IN[followee] / max(self.IN) + 0.5 * (1 - self.IN[follower] / max(self.IN))
        U_popularity = self.IN[followee] / max(self.IN) 
        U_similarity = 1 - abs(self.OPINIONS[follower] - self.OPINIONS[followee])
        #U_proximity = 0.5 * (1 - self.IN[follower] / max(self.IN))
        # 1 / shortestpath, shortest_path is int >= 2
        U_proximity = 1 - (self.SHORTEST_PATH[follower][followee] / max(self.SHORTEST_PATH[follower].values()))


        U_total = (self.w_pop * U_popularity +
                   self.w_sim * U_similarity +
                   self.w_prox * U_proximity)
       
        return U_total
   
    def decide_to_follow(self, follower, followee):
        if self.utility_score(follower,followee) >= np.random.uniform(0,1):
            return True
        else:
            return False


    def decide_to_unfollow(self, follower, followee):
        #U_total = 0.4 * self.WEIGHT[follower][followee] + 0.3 * self.IN[followee]/max(self.IN) + 0.3 * (1 - self.IN[follower]/max(self.IN))
        if self.WEIGHT[follower][followee] <= np.random.uniform(0,1):
        #if U_total <= np.random.uniform(0,1):
            return True
        else:
            return False
   
    def have_encounter_with(self, agent):
        """Use fermi-dirac"""
        agents = list(self.G.nodes())
        agents.remove(agent)  # Removing the agent itself from the list
   
        distances = []
        for followee in self.G.nodes():
            if followee == agent:
                continue
            elif followee in self.SHORTEST_PATH[agent]:
                distances.append(self.SHORTEST_PATH[agent][followee])
            else:
                # have number of weights match the number of agents 
                agents.remove(followee)
            # distances.append(self.SHORTEST_PATH[agent][followee])
        distances = np.array(distances)
        if len(distances) == 0:
            return
        
        exponents = (distances / max(self.SHORTEST_PATH) - self.mu) / self.temperature
        #print(exponents)
        #exponents = (distances / max(distances) - self.mu) / self.temperature
        probs = expit(-exponents)
   
        total_prob = np.sum(probs)
        if total_prob == 0:
            norm_probs = np.zeros_like(probs)
        else:
            norm_probs = probs / total_prob


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
        avg_degree = sum(nx.average_degree_connectivity(self.G).values()) / self.n_agents
        avg_clustering_coeff = nx.average_clustering(self.G)
        centrality = nx.betweenness_centrality(self.G)
        degree_sequence = sorted([d for n, d in self.G.degree()], reverse=True)
        
        
        self.Data_Collector["max IN degrees"].append(max(self.IN.values()))
        self.Data_Collector["avg degrees"].append(avg_degree)
        self.Data_Collector["avg clustering coeff"].append(avg_clustering_coeff)
        self.Data_Collector["betweenness centrality"].append(centrality)
        self.Data_Collector["degree sequence"].append(degree_sequence)
        self.Data_Collector["IN degree"].append(self.IN.values())


    def step(self):
        # initialize list to keep track how the network should change
        edges_to_add = []
        edges_to_remove = []
        
        for node in self.G.nodes():

            encountered = self.have_encounter_with(node)
            if not encountered:
                continue
            # in this case the agent does not follow the potential followee yet
           
            if self.SHORTEST_PATH[node][encountered] > 1:
            # if the agent does not follow the person yet;
            # decide if you want to follow (based on opinion similarity)
                    if self.decide_to_follow(node, encountered):
                        edges_to_add.append((node, encountered))
                       
            # if the agent does already follow, check if you want to keep following
            # based on engagement
            elif self.SHORTEST_PATH[node][encountered] == 1:
                if self.decide_to_unfollow(node, encountered):
                    edges_to_remove.append((node, encountered))
                   
        # for computational reasons first all agents do an action
        # after that the actual network is changed
        for follower, followee in edges_to_add:
            self.add_connection(follower, followee)


        for follower, exfollowee in edges_to_remove:
            self.remove_connection(follower, exfollowee)


        self.update_opinions_and_weights()
        self.track_metrics()
        self.SHORTEST_PATH = dict(nx.shortest_path_length(self.G))


