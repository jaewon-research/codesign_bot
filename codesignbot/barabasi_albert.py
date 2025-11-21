# save as: oasis/generator/twitter/ba_generator.py

import networkx as nx
import pandas as pd
import random
import json

def generate_ba_graph(
    n_agents=100,
    m=5,
    base_csv=None,  # Optional: existing CSV to add graph to
    output_csv='ba_network_agents.csv'
):
    """
    Generate BA graph and create OASIS-compatible CSV.
    
    This integrates directly with existing OASIS workflow.
    """
    
    # 1. Generate BA graph
    print(f"Generating Barabási-Albert graph ({n_agents} agents, m={m})...")
    G = nx.barabasi_albert_graph(n_agents, m)
    
    # 2. Convert to directed (Twitter-style)
    directed_G = nx.DiGraph()
    for edge in G.edges():
        if random.random() < 0.7:
            directed_G.add_edge(edge[0], edge[1])
        if random.random() < 0.3:
            directed_G.add_edge(edge[1], edge[0])
    
    # 3. If base CSV provided, load it and add graph
    if base_csv:
        df = pd.read_csv(base_csv)
        print(f"Loaded existing CSV with {len(df)} agents")
    else:
        # Create simple personas
        df = pd.DataFrame({
            'username': [f'user_{i}' for i in range(n_agents)],
            'name': [f'Agent {i}' for i in range(n_agents)],
            'description': [f'Bio for agent {i}' for i in range(n_agents)],
            'user_char': [f'Persona for agent {i}' for i in range(n_agents)],
        })
    
    # 4. Add following lists
    following_lists = []
    for i in range(n_agents):
        followers = list(directed_G.successors(i))
        following_lists.append(followers)
    
    df['following_agentid_list'] = following_lists
    df['previous_tweets'] = '[]'
    
    # 5. Save
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved to {output_csv}")
    
    return df


if __name__ == "__main__":
    # REPLACE the random graph generation
    generate_ba_graph_for_oasis(
        n_agents=997,  # Same as original
        m=7,           # Controls connectivity
        output_csv='./1k_ba.csv'  # Your output
    )