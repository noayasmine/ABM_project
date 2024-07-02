#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:48:15 2024

@author: noaroebersen
"""
import SALib
from SALib.sample import saltelli
from wolf_sheep.model import WolfSheep
from wolf_sheep.agents import Wolf, Sheep
from mesa.batchrunner import FixedBatchRunner
from SALib.analyze import sobol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm  # For progress bar



# Define parameters and bounds
problem = {
    'num_vars': 3,
    'names': ['sheep_reproduce', 'wolf_reproduce', 'wolf_gain_from_food'],
    'bounds': [[0.01, 0.1], [0.01, 0.1], [5, 30]]
}

# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 10
max_steps = 100
distinct_samples = 10

# We get all our samples here
param_values = saltelli.sample(problem, distinct_samples)


# Define your model
def run_model(params):
    sheep_reproduce, wolf_reproduce, wolf_gain_from_food = params
    # Placeholder for your model
    # Replace with your actual model implementation
    final_sheep_population = np.random.rand()  # Example output
    final_wolf_population = np.random.rand()  # Example output
    return final_sheep_population, final_wolf_population

results = pd.DataFrame(index=range(num_samples), columns=problem['names'] + ['Sheep', 'Wolves'])

# Batch run
count = 0
for i in tqdm(range(replicates)):
    for vals in param_values:
        # Ensure integer parameters are cast correctly if needed
        vals = list(vals)
        vals[2] = int(vals[2])  # Example if 'wolf_gain_from_food' should be integer

        # Run the model with the current set of parameters
        sheep, wolves = run_model(vals)

        # Store results
        results.iloc[count, :3] = vals
        results.iloc[count, 3] = sheep
        results.iloc[count, 4] = wolves

        count += 1

# Save results to CSV
results.to_csv('simulation_results.csv', index=False)