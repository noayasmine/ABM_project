#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors Francijn, Saulo, Noa and Rafael

In this project, we aim to investigate the interplay between
various recommendation strategies and their effects on network polarization
and structural dynamics. Specifically, we focus on how different
combinations of factors, such as popularity, social proximity, and opinion similarity,
in who-to-follow recommendations influence the evolving patterns of opinion formation
and network evolution within a social network. 

Our primary research question is: In an opinion transmission and link creation model
where agents consider popularity, social proximity, and opinion similarity as criteria
to engage with other agents and their opinions, which of these characteristics will yield 
a higher effect on shaping opinion polarization across the network?

This model creates the model and defines the agents, steps and network itself.
If you want to run the model, use single_run.py or batch_run.py.
"""

import numpy as np
import networkx as nx
import random as r
from collections import defaultdict
import networkx.algorithms.community as nx_comm

class SocialNetwork():
    """Simulate a social media network as an agent-based model.
    
    This class uses networkx to initialize a network and keeps track of various metrics
    in dictionaries for computational efficiency.
    """
    
    def __init__(self, n_agents, prob, w_pop, w_prox, w_sim, sociability):
        self.n_agents = n_agents
        self.prob = prob
        self.w_pop = w_pop
        self.w_prox = w_prox
        self.w_sim = w_sim
        self.prob = prob

        if self.w_pop + self.w_sim + self.w_prox != 1:
            print("make sure the weights of w_pop, w_prox and w_sim add to 1")
            raise ValueError
    
        # Initialize dictionaries to keep track of in-degrees, out-degrees, and weights

        self.IN = defaultdict(int)
        self.OUT = defaultdict(int)
        self.WEIGHT = defaultdict(lambda: defaultdict(float))
        self.OPINIONS = {i: r.uniform(0, 1) for i in range(n_agents)}
        self.SOCIABILITY = sociability
        self.SHORTEST_PATH = defaultdict(int)
        self.Data_Collector = {"max IN degrees": [], "avg degrees": [],
                               "avg clustering coeff" : [], "betweenness centrality" : [], 
                               "IN degree": [], "modularity":[]}
        
        self.create_random_network()
   
    def create_random_network(self):
        """Initialize a random network and add bidirectional edges, including edge weights
        (initially drawn from U(1,0)).
        """
        
        # Create Erdos Renyi graph
        self.G = nx.gnp_random_graph(n=self.n_agents, p=self.prob, directed=True)

        # Add weights to the edges and track weights in a dictionary
        for node in self.G.nodes():
            out_edges = list(self.G.out_edges(node))
            
            if out_edges:
                engagements = np.random.uniform(0,1,len(out_edges))
                
                for (u, v), engagement in zip(out_edges, engagements):
                    self.WEIGHT[u][v] = engagement
                    # add in and out degree per node
                    self.OUT[u] += 1
                    self.IN[v] += 1
                    
        # Create a dictionary with the shortest path lengths for quick access
        self.SHORTEST_PATH = dict(nx.shortest_path_length(self.G))
   
    
    def update_opinions_and_weights(self):
        """Update the opinions and the weights of the edges based on the opinions of neighbors."""

        for node in self.G.nodes():
            sociability = self.SOCIABILITY
            opinions = []
            weights = []
            new_weights = []

            for i in self.WEIGHT[node]:
                if abs(self.OPINIONS[i] - self.OPINIONS[node]) > sociability:
                    continue
                opinions.append(self.OPINIONS[i])
                weights.append(self.WEIGHT[node][i])

            if sum(weights) == 0:
                continue
            
            else:
                consensus = np.average(opinions, weights=weights)
                
                old_opinion = self.OPINIONS[node]
                new_opinion = old_opinion + sociability * (consensus - old_opinion) + np.random.normal(0, 0.01)
                
                self.OPINIONS[node] = np.clip(new_opinion, 0, 1)
            
            for i in self.WEIGHT[node]:
                old_diff = abs(self.OPINIONS[i] - old_opinion)
                new_diff = abs(self.OPINIONS[i] - new_opinion)
                
                if new_diff != 0:
                    weight_adjustment = old_diff / new_diff
                else:
                    weight_adjustment = 1.0  # Max weight if new_diff is 0 (opinions are identical)
                
                # Increase the weights a bit for popular agents
                self.WEIGHT[node][i] *= ((1-sociability) + (self.IN[i] / max(self.IN) * sociability))
                noise = np.random.normal(0, 0.01)  # Adding noise to the weight adjustment
                self.WEIGHT[node][i] *= weight_adjustment + noise
                new_weights.append(self.WEIGHT[node][i])
                
            for i in self.WEIGHT[node]:
                new_weight = self.WEIGHT[node][i]
                min_weight = np.min(new_weights) if len(new_weights) != 0 else 0
                max_weight = np.max(new_weights) if len(new_weights) != 0 else 1
                
                if max_weight == 0:
                    max_weight = 1
                    min_weight = 0
                if max_weight - min_weight == 0:
                    continue
                else:
                    self.WEIGHT[node][i] = (new_weight - min_weight + 0.001) / (max_weight - min_weight + 0.001)
               

    def utility_score(self, follower, followee):
        """Calculate the utility score for a follower following a followee.
        
        The utility score is a weighted sum of popularity, similarity, and proximity.
        """
        
        U_popularity = self.IN[followee] / max(self.IN) 
        U_similarity = 1 - abs(self.OPINIONS[follower] - self.OPINIONS[followee])
        U_proximity = 1 - (self.SHORTEST_PATH[follower][followee] / max(self.SHORTEST_PATH[follower].values()))

        U_total = (self.w_pop * U_popularity +
                   self.w_sim * U_similarity +
                   self.w_prox * U_proximity)
       
        return U_total
   
    def decide_to_follow(self, follower, followee):
        """Decide if a follower should follow a followee based on the utility score."""

        if self.utility_score(follower,followee) >= np.random.uniform(0,1):
            return True
        else:
            return False


    def decide_to_unfollow(self, follower, followee):
        """Decide if a follower should unfollow a followee based on the engagement weight."""

        if self.WEIGHT[follower][followee] <= np.random.uniform(0,1):
            return True
        else:
            return False
   
    def have_encounter_with(self, agent):
        """Use Fermi-Dirac distribution to decide which agent will be encountered."""

        agents = list(self.G.nodes())
        agents.remove(agent)  # Remove agent itself from the list
   
        distances = []
        for followee in self.G.nodes():
            if followee == agent:
                continue
            elif followee in self.SHORTEST_PATH[agent]:
                distances.append(self.SHORTEST_PATH[agent][followee])
            else:
                agents.remove(followee)

        distances = np.array(distances)
        if len(distances) == 0:
            return
        
        
        exponents = ((distances / max(self.SHORTEST_PATH) - self.SOCIABILITY) / (0.01 + (0.99 * (1-self.w_prox))))
        probs = 1/(1+np.exp(-exponents))
        
        # Select an agent based on cumulative probabilities
        encounter = r.choices(agents, weights=probs, k=1)
        return encounter[0]
   
    def add_connection(self, follower, followee):
        """Add a connection (edge) between a follower and a followee."""
        
        if not self.G.has_edge(follower, followee):
            self.G.add_edge(follower, followee)
            engagement = r.uniform(0, 1)
            self.WEIGHT[follower][followee] = engagement
            self.OUT[follower] += 1
            self.IN[followee] += 1


    def remove_connection(self, follower, followee):
        """Remove a connection (edge) between a follower and a followee."""
        
        if self.G.has_edge(follower, followee):
            self.G.remove_edge(follower, followee)
            del self.WEIGHT[follower][followee]
            self.OUT[follower] -= 1
            self.IN[followee] -= 1


    def track_metrics(self):
        """Track and collect network metrics such as degree, clustering coefficient, centrality, and modularity."""

        avg_degree = (sum(self.IN.values()) + sum(self.OUT.values())) / self.n_agents
        avg_clustering_coeff = nx.average_clustering(self.G)
        centrality = nx.betweenness_centrality(self.G)            
        communities = nx_comm.greedy_modularity_communities(self.G)
        
        if self.G.number_of_edges() == 0:
            modularity = np.NaN
            max_IN = 0
            
        # Calculate modularity
        else:
            modularity = nx_comm.modularity(self.G, communities)
            max_IN = max(self.IN.values())
        
        self.Data_Collector["max IN degrees"].append(max_IN)
        self.Data_Collector["avg degrees"].append(avg_degree)
        self.Data_Collector["avg clustering coeff"].append(avg_clustering_coeff)
        self.Data_Collector["betweenness centrality"].append(centrality)
        self.Data_Collector["IN degree"].append(self.IN.values())
        self.Data_Collector["modularity"].append(modularity)


    def step(self):
        """Perform a simulation step where agents decide to follow or unfollow others and update their opinions."""

        edges_to_add = []
        edges_to_remove = []
        
        for node in self.G.nodes():
            
            # Encounter other agent
            encountered = self.have_encounter_with(node)
            if not encountered:
                continue

            # Agent does not yet follow encountered agent
            if self.SHORTEST_PATH[node][encountered] > 1:
                    # Decide to follow or not
                    if self.decide_to_follow(node, encountered):
                        edges_to_add.append((node, encountered))
                       
            # Agent already follows encountered agent
            elif self.SHORTEST_PATH[node][encountered] == 1:
                # Decide to unfollow or not
                if self.decide_to_unfollow(node, encountered):
                    edges_to_remove.append((node, encountered))
                   
        # Update network once every time step
        for follower, followee in edges_to_add:
            self.add_connection(follower, followee)

        for follower, exfollowee in edges_to_remove:
            self.remove_connection(follower, exfollowee)

        self.update_opinions_and_weights()
        self.track_metrics()
        self.SHORTEST_PATH = dict(nx.shortest_path_length(self.G))
