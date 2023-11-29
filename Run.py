# DON-CODE
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy.sparse.csgraph import minimum_spanning_tree

def parse_edge_description(description):
    match = re.match(r'(\d+)-(\d+)=>(\d+)', description)
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))
    else:
        raise ValueError(f"Invalid edge description: {description}")

def graph_to_matrix(graph, max_nodes):
    matrix = nx.to_numpy_array(graph)
    matrix = np.pad(matrix, ((0, max_nodes - matrix.shape[0]), (0, max_nodes - matrix.shape[1])))
    return matrix

def predict_minimum_spanning_tree(model, graph_matrix):
    prediction = model.predict(np.array([graph_matrix]))
    return prediction.reshape(graph_matrix.shape)

def visualize_graph_and_mst(graph_matrix, mst_matrix):
    num_nodes = graph_matrix.shape[0]

    # Extract edges and weights from the original graph
    edges = np.array(np.where(graph_matrix > 0)).T
    weights = graph_matrix[graph_matrix > 0]

    # Identify isolated nodes in the original graph
    isolated_nodes_original = set(range(num_nodes)) - set(edges.flatten())

    # Create a figure with two subplots
    plt.figure(figsize=(12, 4))

    # Plot the original graph without isolated nodes
    plt.subplot(121, frame_on=False)
    plt.title("Original Graph")
    nx_graph = nx.from_numpy_array(graph_matrix)
    pos = nx.spring_layout(nx_graph)

    # Create a mapping between entered nodes and NetworkX nodes
    node_mapping = {int(node): i for i, node in enumerate(nx_graph.nodes)}

    # Filter out isolated nodes from the nodes to be drawn
    nodes_to_draw = set(node_mapping.values()) - set([node_mapping[node] for node in isolated_nodes_original])

    # Draw the edges
    nx.draw_networkx_edges(nx_graph, pos, edgelist=edges, edge_color='gray')

    # Draw the nodes without isolated nodes
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=nodes_to_draw, node_size=700, node_color='skyblue')

    # Add labels to nodes without isolated nodes
    labels = {i: str(i) for i in nodes_to_draw}
    nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=8)

    labels = {(i, j): str(int(graph_matrix[i, j])) for i, j in edges}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labels, font_size=8)

    # Extract edges and weights from the predicted MST
    mst = minimum_spanning_tree(graph_matrix).toarray()
    mst_edges = np.array(np.where(mst > 0)).T
    mst_weights = mst[mst > 0]

    # Identify isolated nodes in the MST
    isolated_nodes_mst = set(range(num_nodes)) - set(mst_edges.flatten())

    # Create a mapping between MST nodes and NetworkX nodes
    mst_node_mapping = {node: i for i, node in enumerate(set(mst_edges.flatten()))}

    # Plot the predicted minimum spanning tree without isolated nodes
    plt.subplot(122, frame_on=False)
    plt.title("Predicted Minimum Spanning Tree")
    nx_mst = nx.from_numpy_array(mst)
    pos_mst = nx.spring_layout(nx_mst)

    # Draw the edges
    nx.draw_networkx_edges(nx_mst, pos_mst, edgelist=mst_edges, edge_color='gray')

    # Draw the nodes without isolated nodes
    nx.draw_networkx_nodes(nx_mst, pos_mst, nodelist=list(mst_node_mapping.keys()), node_size=700, node_color='lightcoral')

    # Add labels to nodes without isolated nodes
    labels_mst = {i: str(i) for i in mst_node_mapping.keys()}
    nx.draw_networkx_labels(nx_mst, pos_mst, labels=labels_mst, font_size=8)

    labels_mst = {(i, j): str(int(mst[i, j])) for i, j in mst_edges}
    nx.draw_networkx_edge_labels(nx_mst, pos_mst, edge_labels=labels_mst, font_size=8)

    plt.show()

def main():
    # Load the trained model
    model = keras.models.load_model('mst_prediction_model.h5')

    # Ask the user for graph input
    graph_input = input("Enter the graph in the form of edges with weights (e.g., '1-2=>3, 2-3=>5, 3-4=>2'): ")

    # Remove spaces from user input
    graph_input = graph_input.replace(" ", "")

    # Check if the input is empty
    if not graph_input:
        print("Error: Empty input. Please enter a valid graph.")
        return

    # Create a graph from user input
    user_graph = nx.Graph()
    for edge_description in graph_input.split(','):
        try:
            u, v, weight = parse_edge_description(edge_description)
            user_graph.add_edge(u, v, weight=weight)
        except ValueError as e:
            print(f"Error: {e}. Please enter a valid edge description.")
            return

    # Convert the user graph to an adjacency matrix
    max_nodes = 20  # You can adjust this based on your requirements
    user_graph_matrix = graph_to_matrix(user_graph, max_nodes)

    # Predict the minimum spanning tree
    predicted_mst_matrix = predict_minimum_spanning_tree(model, user_graph_matrix)

    # Visualize the user graph and the predicted minimum spanning tree
    visualize_graph_and_mst(user_graph_matrix, predicted_mst_matrix)

if __name__ == "__main__":
    main()
