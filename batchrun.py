import itertools
from ABM_Final2406_new import SocialNetwork
import pandas as pd


# Define the possible values
values = [0, 0.25, 0.5, 0.75, 1]


# Generate all possible combinations of a, b, c
combinations = []


for a, b, c in itertools.product(values, repeat=3):
    if a + b + c == 1:
        combinations.append((a, b, c))


# so now we have all the combinations we want


# All the other parameters stay the same
steps = 550
n_agents = 100
avg_degree = 50
prob = avg_degree / n_agents


# not sure what this is for exactly
concentration = 2.5


# how important either of these things is (this decides whom to have an interaction with)
k_graph = int(prob)
p_graph = 0.5
mu = 2.0  # Estimated average social distance for connections
temp = 1.0  # Initial guess for temperature


for i in range(len(combinations)):
    print(i)
    w_popularity = combinations[i][0]
    w_proximity = combinations[i][1]
    w_similarity = combinations[i][2]
   
    model = SocialNetwork(n_agents, prob, concentration, w_popularity, w_proximity, w_similarity, mu, temp)
    for j in range(steps + 1):
        model.step()
        print(f"\rProgress: {(j / steps) * 100:.2f}%", end='', flush=True)


    # Save results to CSV
    df_results = pd.DataFrame(model.Data_Collector)
    w_pop_list = [w_popularity] * steps
    w_prox_list = [w_proximity] * steps
    w_sim_list = [w_similarity] * steps
    # runs = [i] * steps
    # df_results.insert(0, "run", runs)
    # df_results.insert(0, "w_pop", w_pop_list)
    # df_results.insert(0, "w_prox", w_prox_list)
    # df_results.insert(0, "w_sim", w_sim_list)
    df_results.to_csv(f"{i}_pop{w_popularity}prox{w_proximity}sim{w_similarity}.csv", index=False)


    end_opinions = pd.DataFrame({'Opinions': model.OPINIONS})
    end_opinions.to_csv(f"{i}_opinions.csv", index=False)