# Explanation of the Network Model

## Network Initialization

The network is initialized as an Erdős–Rényi (ER) directed network $G(N, p)$, where $N$ is the number of nodes and $p$ is the probability of edge creation between any two nodes. Each edge $(u, v)$ in the network has an associated engagement value $e_{uv}$ randomly assigned within the range $[0.1, 0.9]$.

## Classification of Nodes

Nodes in the network are classified as either influencers or followers based on their in-degree. The average in-degree of the network is calculated, and nodes with an in-degree greater than this average are classified as influencers (denoted by $I$), while the others are classified as followers (denoted by $F$).

For each node $n$, additional counters are initialized to keep track of the gains from each strategy:
- `counter_random_influencer`
- `counter_targeted_influencer`
- `counter_most_popular_influencer`
- `counter_random_follower`
- `counter_targeted_follower`
- `counter_most_popular_follower`

## Influencer Strategies

Influencers have three strategies for attracting followers, which involve being added to one of three recommendation pools:
1. **Random Pool**: Influencers are added to this pool randomly.
2. **Targeted Pool**: Influencers are added to this pool based on the targeted recommendation strategy.
3. **Most Popular Pool**: Influencers are added to this pool based on their popularity (in-degree).

## Follower Strategies

Followers have three strategies for receiving recommended influencers from the system:
1. **Random Recommendation**: A follower is recommended a random influencer from the random pool.
2. **Targeted Recommendation**: A follower is recommended an influencer from the targeted pool, with the recommendation probability proportional to the audience fitness of the potential follower in the influencer's follower base. The audience fitness $A_{ij}$ is calculated as:
   
   ![image](https://github.com/noayasmine/ABM_project/assets/71283957/886c8112-221b-4c59-a81c-a12af83fd965)
   
   where $e_{kj}$ is the engagement from follower $k$ to influencer $j$.
4. **Most Popular Recommendation**: A follower is recommended an influencer from the most popular pool, with the recommendation probability proportional to the popularity (in-degree) of that influencer.

## Similarity Index

The similarity index $S_{ij}$ between two nodes $i$ and $j$ is defined to measure the similarity of their influencers:

![image](https://github.com/noayasmine/ABM_project/assets/71283957/99fdd276-1ad5-4a57-ad31-161892fa7cf5)

where $C_i$ and $C_j$ are the sets of common successors (influencers) of $i$ and $j$, respectively.

## Edge Maintenance Decision

After a follower $f$ receives a recommendation by the system of follower $i$, its audience fitness is compared to the audience fitness of the influencers the follower already follows, and the engagement of the one with the closest audience fitness is adopted for the new influencer. Then, the influencer is added to the out-edges of follower $f$ with a probability equal to its engagement. After this, engagements are renormalized.

If the new influencer is followed, the engagement value $e_{fi}$ reinforces the corresponding strategy counter for both the follower and the influencer.

## Simulation Protocol

1. **Network Setup**: The network is created and nodes are classified as influencers or followers.
2. **Strategy Assignment**: Initial strategy pools are filled based on a random selection or specified manual probabilities.
3. **Follower Recommendations**: Followers receive influencer recommendations based on the specified strategies.
4. **Engagement Evaluation**: Followers decide to maintain or remove the new follow based on its engagement.
5. **Edge Removal**: Periodically, edges are removed from the network based on the specified culling interval and scope.
6. **Metrics Calculation**: Network metrics such as average in-degree, average out-degree, average clustering coefficient, and average path length are recorded.

## Network Metrics

Throughout the simulation, various network metrics are recorded and analyzed:
- **Average Out-Degree**: The mean number of outgoing edges per node.
- **Average In-Degree**: The mean number of incoming edges per node.
- **Average Clustering Coefficient**: The average clustering coefficient of the network, reflecting the tendency of nodes to cluster together.
- **Average Path Length**: The average shortest path length between nodes in the network.

## Strategy Usage Analysis

The usage of each strategy (random, targeted, most popular) for both influencers and followers is tracked over time, providing insights into the dynamics and effectiveness of different strategies.

By thoroughly simulating and analyzing these dynamics, the model provides a comprehensive understanding of how influencers and followers interact in a directed network, how engagement evolves, and how different strategies impact network structure and behavior.


## I also added a "free_for_all" mode in which the model is run considering all the agents as influencers and followers equally, just in case you want to test it

