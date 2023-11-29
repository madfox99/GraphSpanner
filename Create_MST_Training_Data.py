# DON-CODE
# This only create a 20x20 tensor, because of that this has a maximum 20 nodes limit

import networkx as nx
import numpy as np
import joblib

def generate_random_graph():
    # Generate a random number of nodes (between 2 and 20 for example)
    num_nodes = np.random.randint(2, 20)
    
    # Generate a random graph with random edge weights
    G = nx.gnm_random_graph(num_nodes, num_nodes * 2)
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 20)  # Assign random weights to edges
    
    return G

def compute_minimum_spanning_tree(graph):
    # Compute the minimum spanning tree using Prim's algorithm
    mst = nx.minimum_spanning_tree(graph)
    return mst

def graph_to_matrix(graph, max_nodes):
    # Convert the graph to an adjacency matrix
    graph_matrix = nx.adjacency_matrix(graph).toarray()
    
    # Pad or truncate the matrix to have a fixed size (max_nodes x max_nodes)
    graph_matrix = np.pad(graph_matrix, ((0, max_nodes - graph_matrix.shape[0]), (0, max_nodes - graph_matrix.shape[1])))
    
    return graph_matrix

def generate_data_and_answers(num_samples, max_nodes=20):
    graph_data = []
    mst_answers = []

    for _ in range(num_samples):
        # Generate a random graph
        graph = generate_random_graph()

        # Compute the minimum spanning tree for the generated graph
        mst = compute_minimum_spanning_tree(graph)

        # Convert the graph and its minimum spanning tree to adjacency matrices
        graph_matrix = graph_to_matrix(graph, max_nodes)
        mst_matrix = graph_to_matrix(mst, max_nodes)

        # Append the data and answers to the lists
        graph_data.append(graph_matrix)
        mst_answers.append(mst_matrix)

    return graph_data, mst_answers

def main():
    # Generate and save at least 50000 datasets
    while True:
        num_samples = input('Number of Samples: ')
        
        # Check if the input consists of digits only
        if num_samples.isdigit():
            break
        else:
            print('Please enter a valid number.')

    # Convert the input to an integer
    num_samples = int(num_samples)

    graph_data, mst_answers = generate_data_and_answers(num_samples)

    # Save the generated graph data and minimum spanning tree answers
    joblib.dump(graph_data, 'graph_data.joblib')
    joblib.dump(mst_answers, 'mst_answers.joblib')

# Check if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    main()
