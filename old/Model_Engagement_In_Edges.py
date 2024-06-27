import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper function to floor values to three decimals
def floor(val, digits):
    factor = 10 ** digits
    return int(val * factor) / factor

# Create the ER directed network with complex engagement assignment
def create_er_directed_network(n, p):
    # n: number of nodes
    # p: probability of edge creation between nodes
    G = nx.gnp_random_graph(n, p, directed=True)
    for node in G.nodes():
        out_edges = list(G.out_edges(node))
        if out_edges:
            engagements = np.random.dirichlet(np.ones(len(out_edges)), size=1)[0]
            engagements = [floor(e, 3) for e in engagements]
            total_engagement = sum(engagements)
            diff = 1 - total_engagement
            if diff != 0:
                idx = random.choice(range(len(engagements)))
                engagements[idx] += diff
            for (u, v), engagement in zip(out_edges, engagements):
                G.edges[u, v]['engagement'] = engagement
    return G

# Assign influencer attribute to nodes
def assign_influencer_attribute(G):
    avg_in_degree = np.mean([d for n, d in G.in_degree()])
    for node in G.nodes():
        G.nodes[node]['influencer'] = 1 if G.in_degree(node) > avg_in_degree else 0
        G.nodes[node]['counter_random_influencer'] = 0.1
        G.nodes[node]['counter_targeted_influencer'] = 0.1
        G.nodes[node]['counter_most_popular_influencer'] = 0.1
        G.nodes[node]['counter_targeted_follower'] = 0.1
        G.nodes[node]['counter_random_follower'] = 0.1
        G.nodes[node]['counter_most_popular_follower'] = 0.1

# Calculate the similarity index between nodes
def similarity_index(G, engagements, i, j, influencers):
    Ci_influencers = set(G.successors(i)) & set(influencers)
    Cj_influencers = set(G.successors(j)) & set(influencers)
    Cij = Ci_influencers & Cj_influencers
    
    if len(Ci_influencers) * len(Cj_influencers) == 0:
        return 0

    sum_convergent = sum(0.9 - abs(engagements[(i, k)] - engagements[(j, k)]) for k in Cij)
    
    return (len(Cij) / (len(Ci_influencers) * len(Cj_influencers))) * sum_convergent

# Normalize a list of values
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5 for _ in values]
    return [0.1 + 0.8 * (val - min_val) / (max_val - min_val) for val in values]

# Remove edges based on a specific culling modality
def remove_edges(G, in_degrees, out_degrees, engagements, percentage, culling_scope, influencers, culling_modality):
    edges = list(G.edges())
    if culling_scope == "influencers":
        edges = [(u, v) for u, v in edges if v in influencers]
    
    num_edges_to_remove = int(percentage * len(edges))
    eligible_edges = [(u, v) for u, v in edges if not (
        (in_degrees[v] == 1 and out_degrees[v] == 0) or 
        (out_degrees[u] == 1 and in_degrees[u] == 0) or 
        (in_degrees[v] == 1 and out_degrees[v] == 1)
    )]
    
    if culling_modality == "strongest_edges_most_likely":
        edge_weights = [engagements[edge] for edge in eligible_edges]
        probabilities = edge_weights / np.sum(edge_weights)
    elif culling_modality == "weakest_edges_most_likely":
        edge_weights = [(1 - engagements[edge]) for edge in eligible_edges]
        probabilities = edge_weights / np.sum(edge_weights)
    else:
        probabilities = [1 / len(eligible_edges)] * len(eligible_edges)

    edge_indices = np.arange(len(eligible_edges))
    edges_to_remove_indices = np.random.choice(edge_indices, size=min(num_edges_to_remove, len(eligible_edges)), p=probabilities, replace=False)
    edges_to_remove = [eligible_edges[i] for i in edges_to_remove_indices]
    
    for u, v in edges_to_remove:
        G.remove_edge(u, v)
        del engagements[(u, v)]
        in_degrees[v] -= 1
        out_degrees[u] -= 1

    for node in G.nodes():
        out_edges = list(G.out_edges(node))
        if out_edges:
            engagements = np.random.dirichlet(np.ones(len(out_edges)), size=1)[0]
            engagements = [floor(e, 3) for e in engagements]
            total_engagement = sum(engagements)
            diff = 1 - total_engagement
            if diff != 0:
                idx = random.choice(range(len(engagements)))
                engagements[idx] += diff
            for (u, v), engagement in zip(out_edges, engagements):
                G.edges[u, v]['engagement'] = engagement

# Calculate probabilities based on counters
def calculate_probs(counter1, counter2, fixed_prob):
    total = counter1 + counter2
    return [(1 - fixed_prob) * (counter1 / total), (1 - fixed_prob) * (counter2 / total)]

# Determine manual probabilities for strategies
def determine_manual_probabilities(manual_random_p, manual_targeted_p, manual_popular_p):
    probs = [manual_random_p, manual_targeted_p, manual_popular_p]
    valid_probs = [p for p in probs if p is not None]
    if any(p is not None and (p < 0 or p > 1) for p in valid_probs):
        raise ValueError("Invalid probability: All manual probabilities must be between 0 and 1.")
    if len(valid_probs) == 0:
        return None
    if len(valid_probs) == 1:
        fixed_prob = valid_probs[0]
        if fixed_prob >= 1:
            raise ValueError("Invalid probability: The provided probability must be less than 1.")
        fixed_idx = probs.index(fixed_prob)
        if fixed_idx == 0:
            return [fixed_prob, *calculate_probs(0.1, 0.1, fixed_prob)]
        elif fixed_idx == 1:
            return [*calculate_probs(0.1, 0.1, fixed_prob), fixed_prob]
        else:
            return [*calculate_probs(0.1, 0.1, fixed_prob), fixed_prob]
    if len(valid_probs) == 2:
        total_fixed = sum(p for p in valid_probs if p is not None)
        if total_fixed >= 1:
            raise ValueError("Invalid probability: The sum of the provided probabilities must be less than 1.")
        missing_idx = probs.index(None)
        probs[missing_idx] = 1 - total_fixed
        return [p / sum(probs) for p in probs]
    if sum(probs) != 1:
        raise ValueError("Invalid probability: The probabilities must sum to 1.")
    return [p / sum(probs) for p in probs]

# Choose an influencer to follow based on similarity
def choose_followed_based_on_similarity(G, follower, targeted_pool, engagements, influencers):
    audience_fits = {}
    for potential_followed in targeted_pool:
        sim_scores = [similarity_index(G, engagements, follower, aud, influencers) * engagements[(aud, potential_followed)] for aud in G.predecessors(potential_followed)]
        audience_fit = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        audience_fits[potential_followed] = audience_fit
    if audience_fits:
        normalized_fits = normalize(list(audience_fits.values()))
        probs = normalized_fits / np.sum(normalized_fits)
        return np.random.choice(list(audience_fits.keys()), p=probs)
    return random.choice(targeted_pool)

# Choose the most popular influencer to follow
def choose_most_popular_followed(most_popular_pool, in_degrees):
    probs = [in_degrees[inf] for inf in most_popular_pool]
    probs = probs / np.sum(probs)
    return np.random.choice(most_popular_pool, p=probs)

# Choose engagement based on similarity
def choose_engagement_based_on_similarity(G, follower, followed, engagements, influencers):
    followed_followers = list(G.predecessors(followed))
    if not followed_followers:
        return random.uniform(0.1, 0.9)
    similarities = [similarity_index(G, engagements, follower, f, influencers) for f in followed_followers]
    normalized_similarities = normalize(similarities)
    probs = normalized_similarities / np.sum(normalized_similarities)
    chosen_follower = np.random.choice(followed_followers, p=probs)
    return engagements[(chosen_follower, followed)]

# Simulate the network dynamics
def simulate(G, steps, setup_turns, edge_removal_percentage, culling_interval, culling_scope, culling_modality, influencer_strategy, follower_strategy,
             manual_influencer_random_p, manual_influencer_targeted_p, manual_influencer_popular_p,
             manual_follower_random_p, manual_follower_targeted_p, manual_follower_popular_p,
             type_of_run):
    if culling_interval > steps:
        raise ValueError("Invalid culling interval: It must be less than or equal to the total number of steps.")
    avg_out_degree = []
    avg_in_degree = []
    avg_clustering_coefficient = []
    avg_path_length = []
    strategy_usage = {
        "random_influencer": [],
        "targeted_influencer": [],
        "most_popular_influencer": [],
        "random_follower": [],
        "targeted_follower": [],
        "most_popular_follower": []
    }
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    engagements = {(u, v): G.edges[u, v]['engagement'] for u, v in G.edges()}
    if influencer_strategy == "manual-set-up":
        influencer_probs = determine_manual_probabilities(manual_influencer_random_p, manual_influencer_targeted_p, manual_influencer_popular_p)
        if influencer_probs is None:
            influencer_strategy = "auto-set-up"
    if follower_strategy == "manual-set-up":
        follower_probs = determine_manual_probabilities(manual_follower_random_p, manual_follower_targeted_p, manual_follower_popular_p)
        if follower_probs is None:
            follower_strategy = "auto-set-up"
    for t in range(steps):
        print(f"\r{(t/steps)*100:.2f}%", end='', flush=True)
        if t > setup_turns and (t - setup_turns) % culling_interval == 0:
            remove_edges(G, in_degrees, out_degrees, engagements, edge_removal_percentage, culling_scope, [n for n in G.nodes() if G.nodes[n]['influencer'] == 1], culling_modality)
        if t >= setup_turns:
            avg_in_deg = np.mean(list(in_degrees.values()))
            for node in G.nodes():
                G.nodes[node]['influencer'] = 1 if in_degrees[node] > avg_in_deg else 0
        
        if type_of_run == "two_tiers":
            influencers = [n for n in G.nodes() if G.nodes[n]['influencer'] == 1]
            non_influencers = [n for n in G.nodes() if G.nodes[n]['influencer'] == 0]
        else:  # free_for_all mode
            influencers = list(G.nodes())
            non_influencers = list(G.nodes())

        targeted_pool = []
        random_pool = []
        most_popular_pool = []
        new_edges = []

        influencers_random_strategy=0
        influencers_targeted_strategy=0
        influencers_most_popular_strategy=0
        
        for influencer in influencers: #Portion that dictates the behaviour of the influencers based on the strategy chosen
            if t < setup_turns:
                strategy = random.choice(["random", "targeted", "most_popular"])
                if strategy == "random":
                    random_pool.append(influencer)
                elif strategy == "targeted":
                    targeted_pool.append(influencer)
                else:
                    most_popular_pool.append(influencer)
            else:
                if influencer_strategy == "only-random":
                    random_pool.append(influencer)
                    influencers_random_strategy+=1
                elif influencer_strategy == "only-targeted":
                    targeted_pool.append(influencer)
                    influencers_targeted_strategy+=1
                elif influencer_strategy == "only-popular":
                    most_popular_pool.append(influencer)
                    influencers_most_popular_strategy+=1
                elif influencer_strategy == "manual-set-up":
                    rand_val = random.random()
                    if rand_val < influencer_probs[0]:
                        random_pool.append(influencer)
                        influencers_random_strategy+=1
                    elif rand_val < influencer_probs[0] + influencer_probs[1]:
                        targeted_pool.append(influencer)
                        influencers_targeted_strategy+=1
                    else:
                        most_popular_pool.append(influencer)
                        influencers_most_popular_strategy+=1
                else:
                    counter_rand = G.nodes[influencer]['counter_random_influencer']
                    counter_targ = G.nodes[influencer]['counter_targeted_influencer']
                    counter_pop = G.nodes[influencer]['counter_most_popular_influencer']
                    prob_rand = counter_rand / (counter_rand + counter_targ + counter_pop)
                    prob_targ = counter_targ / (counter_rand + counter_targ + counter_pop)
                    rand_val = random.random()
                    if rand_val < prob_rand:
                        random_pool.append(influencer)
                        influencers_random_strategy+=1
                    elif rand_val < prob_rand + prob_targ:
                        targeted_pool.append(influencer)
                        influencers_targeted_strategy+=1
                    else:
                        most_popular_pool.append(influencer)
                        influencers_most_popular_strategy+=1

        
        non_influencers_random_strategy=0
        non_influencers_targeted_strategy=0
        non_influencers_most_popular_strategy=0

        
        for follower in non_influencers: #Portion that dictates the behaviour of the non-influencers based on the strategy chosen
            followed=None
            if t < setup_turns:
                strategy = random.choice(["random", "targeted", "most_popular"])
                if strategy == "random" and random_pool:
                    followed = random.choice(random_pool)
                elif strategy == "targeted" and targeted_pool:
                    followed = choose_followed_based_on_similarity(G, follower, targeted_pool, engagements, influencers)
                elif strategy == "most_popular" and most_popular_pool:
                    followed = choose_most_popular_followed(most_popular_pool, in_degrees)
                else:
                    continue
            else:
                if follower_strategy == "only-random":
                    if random_pool:
                        random_pool_filtered = [node for node in random_pool if node != follower]
                        if not random_pool_filtered:
                            continue
                        followed = random.choice(random_pool_filtered)
                        non_influencers_random_strategy+=1
                    else:
                        continue
                    strategy = "random"
                elif follower_strategy == "only-targeted":
                    if targeted_pool:
                        targeted_pool_filtered = [node for node in targeted_pool if node != follower]
                        if not targeted_pool_filtered:
                            continue
                        followed = choose_followed_based_on_similarity(G, follower, targeted_pool_filtered, engagements, influencers)
                        non_influencers_targeted_strategy+=1
                    else:
                        continue
                    strategy = "targeted"
                elif follower_strategy == "only-popular":
                    if most_popular_pool:
                        most_popular_pool_filtered = [node for node in most_popular_pool if node != follower]
                        if not most_popular_pool_filtered:
                            continue
                        followed = choose_most_popular_followed(most_popular_pool_filtered, in_degrees)
                        non_influencers_most_popular_strategy+=1
                    else:
                        continue
                    strategy = "most_popular"
                elif follower_strategy == "manual-set-up":
                    rand_val = random.random()
                    if rand_val < follower_probs[0] and random_pool:
                        random_pool_filtered = [node for node in random_pool if node != follower]
                        if not random_pool_filtered:
                            continue
                        followed = random.choice(random_pool_filtered)
                        non_influencers_random_strategy+=1
                        strategy = "random"
                    elif rand_val < follower_probs[0] + follower_probs[1] and targeted_pool:
                        targeted_pool_filtered = [node for node in targeted_pool if node != follower]
                        if not targeted_pool_filtered:
                            continue
                        followed = choose_followed_based_on_similarity(G, follower, targeted_pool_filtered, engagements, influencers)
                        non_influencers_targeted_strategy+=1
                        strategy = "targeted"
                    elif most_popular_pool:
                        most_popular_pool_filtered = [node for node in most_popular_pool if node != follower]
                        if not most_popular_pool_filtered:
                            continue
                        followed = choose_most_popular_followed(most_popular_pool_filtered, in_degrees)
                        non_influencers_most_popular_strategy+=1
                        strategy = "most_popular"
                    else:
                        continue
                else:
                    counter_rand = G.nodes[follower]['counter_random_follower']
                    counter_targ = G.nodes[follower]['counter_targeted_follower']
                    counter_pop = G.nodes[follower]['counter_most_popular_follower']
                    prob_rand = counter_rand / (counter_rand + counter_targ + counter_pop)
                    prob_targ = counter_targ / (counter_rand + counter_targ + counter_pop)
                    rand_val = random.random()
                    if rand_val < prob_rand and random_pool:
                        random_pool_filtered = [node for node in random_pool if node != follower]
                        if not random_pool_filtered:
                            continue
                        followed = random.choice(random_pool_filtered)
                        non_influencers_random_strategy+=1
                        strategy = "random"
                    elif rand_val < prob_rand + prob_targ and targeted_pool:
                        targeted_pool_filtered = [node for node in targeted_pool if node != follower]
                        if not targeted_pool_filtered:
                            continue
                        followed = choose_followed_based_on_similarity(G, follower, targeted_pool_filtered, engagements, influencers)
                        non_influencers_targeted_strategy+=1
                        strategy = "targeted"
                    elif most_popular_pool:
                        most_popular_pool_filtered = [node for node in most_popular_pool if node != follower]
                        if not most_popular_pool_filtered:
                            continue
                        followed = choose_most_popular_followed(most_popular_pool_filtered, in_degrees)
                        non_influencers_most_popular_strategy+=1
                        strategy = "most_popular"
                    else:
                        continue

            if followed: #If a potential influencer is recommended, it is decided if it's followed based on probability equals to its engagement
                engagement_val = choose_engagement_based_on_similarity(G, follower, followed, engagements, influencers)
                if random.random() < engagement_val:
                    new_edges.append((follower, followed, engagement_val))
                    G.add_edge(follower, followed)
                    G[follower][followed]['engagement'] = engagement_val
                    engagements[(follower, followed)] = engagement_val
                    in_degrees[followed] += 1
                    out_degrees[follower] += 1
                    out_edges = list(G.out_edges(follower))
                    if out_edges:
                        engagements_list = [engagements[edge] for edge in out_edges]
                        normalized_engagements = np.random.dirichlet(np.ones(len(out_edges)), size=1)[0]
                        normalized_engagements = [floor(e, 3) for e in normalized_engagements]
                        total_engagement = sum(normalized_engagements)
                        diff = 1 - total_engagement
                        if diff != 0:
                            idx = random.choice(range(len(normalized_engagements)))
                            normalized_engagements[idx] += diff
                        for (u, v), new_engagement in zip(out_edges, normalized_engagements):
                            G.edges[u, v]['engagement'] = new_engagement
                            engagements[(u, v)] = new_engagement
                if strategy == "random":
                    G.nodes[follower]['counter_random_follower'] += engagement_val
                    G.nodes[followed]['counter_random_influencer'] += engagement_val
                elif strategy == "targeted":
                    G.nodes[follower]['counter_targeted_follower'] += engagement_val
                    G.nodes[followed]['counter_targeted_influencer'] += engagement_val
                else:
                    G.nodes[follower]['counter_most_popular_follower'] += engagement_val
                    G.nodes[followed]['counter_most_popular_influencer'] += engagement_val

        avg_out_degree.append(np.mean(list(out_degrees.values())))
        avg_in_degree.append(np.mean(list(in_degrees.values())))
        avg_clustering_coefficient.append(nx.average_clustering(G.to_undirected()))
        avg_path_length.append(nx.average_shortest_path_length(G.to_undirected()))
        if t > setup_turns:
            num_influencers = len(influencers)
            if num_influencers > 0:
                strategy_usage["random_influencer"].append(influencers_random_strategy / num_influencers)
                strategy_usage["targeted_influencer"].append(influencers_targeted_strategy / num_influencers)
                strategy_usage["most_popular_influencer"].append(influencers_most_popular_strategy / num_influencers)
            num_non_influencers = len(non_influencers)
            if num_non_influencers > 0:
                strategy_usage["random_follower"].append(non_influencers_random_strategy / num_non_influencers)
                strategy_usage["targeted_follower"].append(non_influencers_targeted_strategy / num_non_influencers)
                strategy_usage["most_popular_follower"].append(non_influencers_most_popular_strategy / num_non_influencers)
        else:
            strategy_usage["random_influencer"].append(0)
            strategy_usage["targeted_influencer"].append(0)
            strategy_usage["most_popular_influencer"].append(0)
            strategy_usage["random_follower"].append(0)
            strategy_usage["targeted_follower"].append(0)
            strategy_usage["most_popular_follower"].append(0)
            
    return avg_out_degree, avg_in_degree, avg_clustering_coefficient, avg_path_length, strategy_usage

# Initialize the network
num_nodes = 100  # Number of nodes in the network
edge_prob = 0.3  # Probability of edge creation between nodes
steps = 500 # Total number of simulation steps
setup_turns = 10  # Number of setup turns before recording metrics
culling_interval = 5  # Interval at which edges are removed
edge_removal_percentage = 0.05  # Percentage of edges to remove during culling
culling_scope = "all"  # Scope of edge removal ("influencers" or "all")
culling_modality = "random"  # Modality for edge removal ("strongest_edges_most_likely", "weakest_edges_most_likely", "random")

influencer_strategy = "auto-set-up"  # Strategy for influencer selection ("only-random", "only-targeted", "only-popular", "manual-set-up", "auto-set-up")
manual_influencer_random_p = None  # Probability for random influencer strategy if manual setup is chosen (between 0 and 1, or None)
manual_influencer_targeted_p = None  # Probability for targeted influencer strategy if manual setup is chosen (between 0 and 1, or None)
manual_influencer_popular_p = None  # Probability for most popular influencer strategy if manual setup is chosen (between 0 and 1, or None)
#If these three probabilities are pre-determined, they must add to 1
follower_strategy = "auto-set-up"  # Strategy for follower selection ("only-random", "only-targeted", "only-popular", "manual-set-up", "auto-set-up")
manual_follower_random_p = None  # Probability for random follower strategy if manual setup is chosen (between 0 and 1, or None)
manual_follower_targeted_p = None  # Probability for targeted follower strategy if manual setup is chosen (between 0 and 1, or None)
manual_follower_popular_p = None  # Probability for most popular follower strategy if manual setup is chosen (between 0 and 1, or None)
#If these three probabilities are pre-determined, they must add to 1
type_of_run = "two_tiers"  # Type of simulation run ("two_tiers" separates the agents in or "free_for_all")

network = create_er_directed_network(num_nodes, edge_prob)
assign_influencer_attribute(network)

avg_out_deg, avg_in_deg, avg_clustering_coef, avg_path_len, strategy_usage = simulate(network, steps, setup_turns, edge_removal_percentage,
                                                                                      culling_interval, culling_scope, culling_modality, influencer_strategy, follower_strategy,
                                                                                      manual_influencer_random_p, manual_influencer_targeted_p, manual_influencer_popular_p,
                                                                                      manual_follower_random_p, manual_follower_targeted_p, manual_follower_popular_p,
                                                                                      type_of_run)

results = {
    "Average Out-Degree": avg_out_deg,
    "Average In-Degree": avg_in_deg,
    "Average Clustering Coefficient": avg_clustering_coef,
    "Average Path Length": avg_path_len
}

df_results = pd.DataFrame(results)
print(df_results)

fig, axs = plt.subplots(6, 1, figsize=(10, 30))

axs[0].plot(df_results.index, df_results["Average Out-Degree"], label='Average Out-Degree')
axs[0].set_title('Average Out-Degree over Time')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Average Out-Degree')
axs[0].legend()

axs[1].plot(df_results.index, df_results["Average In-Degree"], label='Average In-Degree', color='orange')
axs[1].set_title('Average In-Degree over Time')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Average In-Degree')
axs[1].legend()

axs[2].plot(df_results.index, df_results["Average Clustering Coefficient"], label='Average Clustering Coefficient', color='green')
axs[2].set_title('Average Clustering Coefficient over Time')
axs[2].set_xlabel('Time Step')
axs[2].set_ylabel('Clustering Coefficient')
axs[2].legend()

axs[3].plot(df_results.index, df_results["Average Path Length"], label='Average Path Length', color='red')
axs[3].set_title('Average Path Length over Time')
axs[3].set_xlabel('Time Step')
axs[3].set_ylabel('Path Length')
axs[3].legend()

axs[4].plot(df_results.index, strategy_usage["random_influencer"], label='Random Influencer', color='blue')
axs[4].plot(df_results.index, strategy_usage["targeted_influencer"], label='Targeted Influencer', color='purple')
axs[4].plot(df_results.index, strategy_usage["most_popular_influencer"], label='Most Popular Influencer', color='cyan')
axs[4].set_title('Influencer Strategy Usage over Time')
axs[4].set_xlabel('Time Step')
axs[4].set_ylabel('Usage Proportion')
axs[4].legend()

axs[5].plot(df_results.index, strategy_usage["random_follower"], label='Random Follower', color='blue')
axs[5].plot(df_results.index, strategy_usage["targeted_follower"], label='Targeted Follower', color='purple')
axs[5].plot(df_results.index, strategy_usage["most_popular_follower"], label='Most Popular Follower', color='cyan')
axs[5].set_title('Follower Strategy Usage over Time')
axs[5].set_xlabel('Time Step')
axs[5].set_ylabel('Usage Proportion')
axs[5].legend()

plt.tight_layout()
plt.show()
