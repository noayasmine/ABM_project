import itertools
from ABM_Final_Final_Final import SocialNetwork
import pandas as pd
import os

combinations = []

# Iterate over possible values of a, b, c in increments of 0.1
for a in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
    for b in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
        for c in range(0, 11):  # This corresponds to 0 to 1 in steps of 0.1
            if a + b + c == 10:  # We check if the sum is 10 (which corresponds to 1.0 in the original scale)
                combinations.append((a / 10, b / 10, c / 10))  # Convert back to original scale


n_agents = 75
prob = 0.5
sociability = 0.1
steps = 500


# Define the new folder name
output_folder = 'simulation_results'

# Create the directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

n_repeats = 5  # Number of repetitions for each combination

for i in range(len(combinations)):
    print(f"Combination {i+1}/{len(combinations)}")
    w_pop = combinations[i][0]
    w_prox = combinations[i][1]
    w_sim = combinations[i][2]

    for repeat in range(n_repeats):
        print(f"  Repeat {repeat+1}/{n_repeats}")
        # Initialize the model
        model = SocialNetwork(n_agents, prob, w_pop, w_prox, w_sim, sociability)
        
        for k in range(steps + 1):
            model.step()
            print(f"\rProgress: {(k / steps) * 100:.2f}%", end='', flush=True)
        
        # Collect results for the current repeat
        df_results = pd.DataFrame(model.Data_Collector)
        
        # Save the results to a CSV file
        results_filename = f"pop{w_pop}prox{w_prox}sim{w_sim}_repeat{repeat + 1}.csv"
        df_results.to_csv(os.path.join(output_folder, results_filename), index=False)

        # Collect end opinions for the current repeat
        end_opinions = pd.DataFrame({
            'Opinions': model.OPINIONS
        })
        
        # Save the end opinions to a CSV file
        opinions_filename = f"pop{w_pop}prox{w_prox}sim{w_sim}_opinions_repeat{repeat + 1}.csv"
        end_opinions.to_csv(os.path.join(output_folder, opinions_filename), index=False)