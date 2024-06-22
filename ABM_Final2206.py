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
        self.OPINIONS = {i: r.uniform(0, 1) for i in range(n_agents)}
        
        self.SHORTEST_PATH = defaultdict(int)

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
        self.SHORTEST_PATH = dict(nx.shortest_path_length(self.G))
    
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
        options = self.SHORTEST_PATH[unique_id]

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
        
    def choose_to_follow_option2(self, unique_id):
        """Choose a candidate to follow based on shortest path probabilities and opinion similarity."""
        candidates = self.SHORTEST_PATH[unique_id]

        # Filter candidates with shortest path > 1 (not already directly connected)
        candidates = {key: value for key, value in candidates.items() if value not in {0, 1}}

        if not candidates:
            return None  # No valid candidates found

        # Calculate probabilities based on shortest path lengths and opinion similarity
        probabilities = []
        for candidate, shortest_path_length in candidates.items():

            path_similarity = 1 / (shortest_path_length + 1)
            opinion_similarity = (1 / (1 - np.exp(-1))) * (
                np.exp(-abs(self.OPINIONS[unique_id] - self.OPINIONS[candidate])) - np.exp(-1))
            total_similarity = path_similarity * opinion_similarity
            probabilities.append(total_similarity)

        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob == 0:
            return None  # No valid probabilities

        probabilities = [p / total_prob for p in probabilities]

        # Choose a candidate based on the calculated probabilities
        chosen_candidate = r.choices(list(candidates.keys()), weights=probabilities, k=1)[0]
        return chosen_candidate
        
    def choose_to_unfollow(self, unique_id):
        """Choose a random follower to unfollow."""
        followees = list(self.G.successors(unique_id))
        if followees:
            return r.choice(followees)
        return None

    def add_connection(self, follower, followee):
        """Add a directed edge between follower and followee."""
        if not self.G.has_edge(follower, followee):
            self.G.add_edge(follower, followee)
            engagement = r.uniform(0, 1)
            self.WEIGHT[follower][followee] = engagement
            self.UTILITIES[followee][follower] = engagement
            self.OUT[follower] += 1
            self.IN[followee] += 1

    def remove_connection(self, follower, followee):
        """Remove a directed edge between follower and followee."""
        if self.G.has_edge(follower, followee):
            self.G.remove_edge(follower, followee)
            del self.WEIGHT[follower][followee]
            del self.UTILITIES[followee][follower]
            self.OUT[follower] -= 1
            self.IN[followee] -= 1

    def track_metrics(self):
        """Track metrics such as average degrees and clustering coefficient."""
        avg_in_degree = sum(self.IN.values()) / self.n_agents
        avg_out_degree = sum(self.OUT.values()) / self.n_agents
        avg_clustering_coeff = nx.average_clustering(self.G)
        avg_utility = sum(sum(inner_dict.values()) for inner_dict in self.UTILITIES.values()) / self.n_agents

        self.Data_Collector["avg IN degrees"].append(avg_in_degree)
        self.Data_Collector["avg OUT degrees"].append(avg_out_degree)
        self.Data_Collector["avg clustering coeff"].append(avg_clustering_coeff)
        self.Data_Collector["avg utility"].append(avg_utility)

    def step(self):
        """Execute one simulation step."""
        edges_to_add = []
        edges_to_remove = []

        for node in self.G.nodes():
            if self.IN[node] > self.OUT[node]:
                followee = self.choose_to_follow_option2(node)
                if followee is not None:
                    edges_to_add.append((node, followee))
            elif self.IN[node] < self.OUT[node]:
                exfollowee = self.choose_to_unfollow(node)
                if exfollowee is not None:
                    edges_to_remove.append((node, exfollowee))

        for follower, followee in edges_to_add:
            self.add_connection(follower, followee)

        for follower, exfollowee in edges_to_remove:
            self.remove_connection(follower, exfollowee)

        self.normalize_weights()
        self.track_metrics()

        # Update shortest paths after making changes
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
fig, axs = plt.subplots(4, 1, figsize=(10, 20))

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

axs[3].plot(df_results.index, df_results["avg utility"], label='Average Utility', color='blue')
axs[3].set_title('Average Utility over Time')
axs[3].set_xlabel('Time Step')
axs[3].set_ylabel('Utility')
axs[3].legend()

plt.tight_layout()
plt.show()








            

 