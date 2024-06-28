from ABM_Final_Final import SocialNetwork
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

""""
File to do a single run of the model. 
First initialize parameters, then it runs the model and show some data.
"""
# Parameters
steps = 1250
n_agents = 75
avg_degree = 25
prob = avg_degree / n_agents


# how important either of these things is (this decides whom to have an interaction with)
w_popularity = 0.6
w_proximity = 0.2
w_similarity = 0.2
k_graph = int(prob)
p_graph = 0.5
mu = 2.0  # Estimated average social distance for connections
temp = 1.0  # Initial guess for temperature

# Initialize and run the model
model = SocialNetwork(n_agents, prob, w_popularity, w_proximity, w_similarity, mu, temp)

for i in range(steps + 1):
    model.step()
    print(f"\rProgress: {(i / steps) * 100:.2f}%", end='', flush=True)


#end_opinions = pd.DataFrame({'Opinions': model.OPINIONS})
#end_opinions.to_csv("end_opinions.csv", index=False)

# Save results to CSV
df_results = pd.DataFrame(model.Data_Collector)
#df_results.to_csv("social_network_metrics.csv", index=False)


# Plotting metrics
fig, axs = plt.subplots(4, 1, figsize=(5, 20))

# TODO I don't know what the best way to plot this is yet
b_centrality = list(df_results["betweenness centrality"][100].values())
axs[0].plot(range(len(b_centrality)), b_centrality, "bo", label='betweenness centrality time step 1000')
#axs[0].set_title('Average Out-Degree over Time')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('betweenness centrality')
axs[0].legend()


axs[1].plot(df_results.index, df_results["avg degrees"], label='Average Degree', color='orange')
#axs[1].set_title('Average In-Degree over Time')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Average Degree')
axs[1].legend()


axs[2].plot(df_results.index, df_results["avg clustering coeff"], label='Average Clustering Coefficient', color='green')
#axs[2].set_title('Average Clustering Coefficient over Time')
axs[2].set_xlabel('Time Step')
axs[2].set_ylabel('Clustering Coefficient')
axs[2].legend()


axs[3].plot(df_results.index, df_results["max IN degrees"], label='Max In-Degree', color='blue')
#axs[3].set_title('Average Utility over Time')
axs[3].set_xlabel('Time Step')
axs[3].set_ylabel('Max In-Degree')
axs[3].legend()


#plt.subplots_adjust(hspace=5.0)
plt.tight_layout()
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
#plot_degree_distribution(model.G)
