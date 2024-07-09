# ABM_project Group 4 - Opinion Opinion Polarization and the Rise of Influencers in an Agent-based Model

### Authors: Rafael Cardenas Heredia, Francijn Keur, Salo Lacunes, Noa Roebersen

In this project, we aimed to investigate the interplay between various recommendation strategies and their effects on network polarization and structural dynamics. Specifically, we focused on how different combinations of factors, such as popularity, social proximity, and opinion similarity, in who-to-follow recommendations influence the evolving patterns of opinion formation and network evolution within a social network. Our primary research question was: In an opinion transmission and link creation model where agents consider popularity, social proximity, and opinion similarity as criteria to engage with other agents and their opinions, which of these characteristics will yield a higher effect on shaping opinion polarization across the network?

The model itself is in the ABM_Final_Final_Final.py file.

You can simulate a single run by the single_run.py file. First choose the values of the parameters n_agents, steps, prob, sociability, w_sim, w_pop, w_prox where the sum of the last values should add to 1.

A batch run can be done by the batchrun_steps0.1.py file. In this file, you can run the same experiments as we did; where we varied the values for w_sim, w_pop, and w_prox in steps of 0.1, making sure they always added up to 1. Furthermore, you can decide for how many repetitions you would like to run these parameter combinations.

The ternary plots can be made using the merging_data_ternaryplots.ipynb, where first the data is merged and then the repetitions are averaged and finally the ternary plots for clustering coefficient and modularity are made.

The fermi-dirac file was used to get insight into how the probability distribution was for certain values for mu and T.

Lastly, the Sobol Sensitivity Analysis can be done by using the SA_sobol_ABM_parallel.ipynb file. 
