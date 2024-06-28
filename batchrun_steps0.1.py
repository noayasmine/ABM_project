import itertools
from ABM_Final_Final import SocialNetwork
import pandas as pd


# Define the possible values
values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


combinations = []

# Iterate over possible values of a, b, c in increments of 0.1
for a in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
    for b in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
        for c in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
            if a + b + c == 10:  # We check if the sum is 10 (which corresponds to 1.0 in the original scale)
                combinations.append((a / 10, b / 10, c / 10))  # Convert back to original scale


# so now we have all the combinations we want


# All the other parameters stay the same for all the runs
steps = 1250
n_agents = 75
avg_degree = 25
prob = avg_degree / n_agents


# how important either of these things is (this decides whom to have an interaction with)
k_graph = int(prob)
p_graph = 0.5
mu = 2.0  # Estimated average social distance for connections
temp = 1.0  # Initial guess for temperature

# since in the end we want to take the average of

for i in range(len(combinations)):
    print(i)
    w_popularity = combinations[i][0]
    w_proximity = combinations[i][1]
    w_similarity = combinations[i][2]

    model = SocialNetwork(n_agents, prob, w_popularity, w_proximity, w_similarity, mu, temp)
    for k in range(steps + 1):
        model.step()
        print(f"\rProgress: {(k / steps) * 100:.2f}%", end='', flush=True)


    # Save results to CSV
    df_results = pd.DataFrame(model.Data_Collector)
    w_pop_list = [w_popularity] * (steps + 1)
    w_prox_list = [w_proximity] * (steps + 1)
    w_sim_list = [w_similarity] * (steps + 1)
    run = [4] * (steps + 1)
    df_results.insert(0, "run", run)
    df_results.insert(0, "w_pop", w_pop_list)
    df_results.insert(0, "w_prox", w_prox_list)
    df_results.insert(0, "w_sim", w_sim_list)
    df_results.to_csv(f"{4}_pop{w_popularity}prox{w_proximity}sim{w_similarity}_moresteps.csv", index=False)


    end_opinions = pd.DataFrame({'Opinions': model.OPINIONS})
    end_opinions.to_csv(f"{4}_opinions.csv", index=False)
