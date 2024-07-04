import itertools
from ABM_Final_Final import SocialNetwork
import pandas as pd
import os




combinations = []

# Iterate over possible values of a, b, c in increments of 0.1
for a in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
    for b in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
        for c in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
            if a + b + c == 10:  # We check if the sum is 10 (which corresponds to 1.0 in the original scale)
                combinations.append((a / 10, b / 10, c / 10))  # Convert back to original scale


n_agents = 100
prob = 0.5
sociability = 0.09
mu = 0.3
temp = 0.1
steps = 800


# Define the new folder name
output_folder = 'simulation_results'

# Create the directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

n_repeats = 20  # Number of repetitions for each combination

for i in range(len(combinations)):
    print(f"Combination {i+1}/{len(combinations)}")
    w_pop = combinations[i][0]
    w_prox = combinations[i][1]
    w_sim = combinations[i][2]
    
    combined_results = pd.DataFrame()  # DataFrame to store all repetitions
    combined_opinions = pd.DataFrame()  # DataFrame to store end opinions for all repetitions

    for repeat in range(n_repeats):
        print(f"  Repeat {repeat+1}/{n_repeats}")
        # Initialize the model
        model = SocialNetwork(n_agents, prob, w_pop, w_prox, w_sim, mu, temp, sociability)
        
        for k in range(steps + 1):
            model.step()
            print(f"\rProgress: {(k / steps) * 100:.2f}%", end='', flush=True)
        
        # Collect results for the current repeat
        df_results = pd.DataFrame(model.Data_Collector)
        w_pop_list = [w_pop] * (steps + 1)
        w_prox_list = [w_prox] * (steps + 1)
        w_sim_list = [w_sim] * (steps + 1)
        run = [repeat + 1] * (steps + 1)
        df_results.insert(0, "run", run)
        df_results.insert(0, "w_pop", w_pop_list)
        df_results.insert(0, "w_prox", w_prox_list)
        df_results.insert(0, "w_sim", w_sim_list)
        
        combined_results = pd.concat([combined_results, df_results], ignore_index=True)
        
        # Collect end opinions for the current repeat
        end_opinions = pd.DataFrame({'Opinions': model.OPINIONS, 'run': [repeat + 1] * len(model.OPINIONS),
                                     'w_pop': [w_pop] * len(model.OPINIONS), 'w_prox': [w_prox] * len(model.OPINIONS),
                                     'w_sim': [w_sim] * len(model.OPINIONS)})
        combined_opinions = pd.concat([combined_opinions, end_opinions], ignore_index=True)
    
    # Save the combined results to a single file
    combined_results.to_csv(os.path.join(output_folder, f"pop{w_pop}prox{w_prox}sim{w_sim}_20repeats.csv"), index=False)

    # Save the combined end opinions to a single file
    combined_opinions.to_csv(os.path.join(output_folder, f"pop{w_pop}prox{w_prox}sim{w_sim}_opinions.csv"), index=False)
