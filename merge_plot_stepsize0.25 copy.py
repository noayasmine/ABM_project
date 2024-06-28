import pandas as pd
import itertools
import  ternary
import matplotlib.pyplot as plt


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

# we want to have all the titles to be able to merge the files
titles = []
for j in range(len(4)):
    for i in range(len(combinations)):
        a = combinations[i][0]
        b = combinations[i][1]
        c = combinations[i][2]
        titles.append(f"{j}_pop{a}prox{b}sim{c}.csv")


n_step = 1250

col_names = ["w_sim", "w_prox", "w_pop", "run", "max IN degrees", "avg IN degrees", "avg OUT degrees", "avg clustering coeff"]
df_merged = pd.DataFrame(columns=col_names)
merged_df_time_step = []
for i in range(len(titles)):
    df = pd.read_csv(f"{titles[i]}")
    df_merged.loc[i] = df.loc[n_step]


# for the ternary plot
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
tax.boundary(linewidth=0, scale=4)
tax.gridlines(color="black", multiple=4)

# Set axis labels
fontsize = 10
offset = 0.15
tax.left_axis_label("w_pop", fontsize=fontsize, offset=offset)
tax.right_axis_label("w_prox", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("w_sim", fontsize=fontsize, offset=offset)
tax.ticks(axis='lbr', ticks=values, multiple=4, linewidth=0.1, tick_formats="%.2f", offset=0.02)

# Set title
plt.title('Average Clustering Coefficient Heatmap on Ternary Plot', fontsize=10)

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()

# Show the plot
plt.show()