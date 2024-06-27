import pandas as pd
import itertools
import  ternary
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Define the possible values
values = [0, 0.25, 0.5, 0.75, 1]

combinations = []

for a, b, c in itertools.product(values, repeat=3):
    if a + b + c == 1:
        combinations.append((a, b, c))

# we want to have all the titles to be able to merge the files
titles = []
for i in range(len(combinations)):
    a = combinations[i][0]
    b = combinations[i][1]
    c = combinations[i][2]
    titles.append(f"{i}_pop{a}prox{b}sim{c}.csv")

n_step = 800

col_names = ["w_sim", "w_prox", "w_pop", "run", "max IN degrees", "avg IN degrees", "avg OUT degrees", "avg clustering coeff"]
df_merged = pd.DataFrame(columns=col_names)
merged_df_time_step = []
for i in range(len(titles)):
    df = pd.read_csv(f"{titles[i]}")
    df_merged.loc[i] = df.loc[n_step]

heatmap_data = {}
for _, row in df_merged.iterrows():
    # Convert weights to percentage format (assuming the scale is 100)
    w_pop, w_prox, w_sim = row['w_pop']*10, row['w_prox']*10, row['w_sim']*10
    avg_clustering_coeff = row['avg clustering coeff']
    heatmap_data[(w_pop, w_prox, w_sim)] = avg_clustering_coeff


# Create the ternary plot
scale = 10
figure, tax = ternary.figure(scale=scale)

# Draw the heatmap
tax.heatmap(heatmap_data, style="hexagonal", cmap='viridis', colorbar=True)

# Draw boundary and gridlines
tax.boundary(linewidth=2.0)
tax.gridlines(color="black", multiple=10)

# Set axis labels
fontsize = 15
tax.left_axis_label("w_pop", fontsize=10)
tax.right_axis_label("w_prox", fontsize=10)
tax.bottom_axis_label("w_sim", fontsize=10)

# Set title
plt.title('Average Clustering Coefficient Heatmap on Ternary Plot', fontsize=10)

# Show the plot
tax.clear_matplotlib_ticks()
plt.show()
