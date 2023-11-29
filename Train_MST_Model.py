# DON-CODE
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def visualize_graph_and_mst(graph_matrix, mst_matrix):
    # Create NetworkX graph from the adjacency matrix
    graph = nx.Graph()
    graph.add_weighted_edges_from(get_edges_from_matrix(graph_matrix))

    mst = nx.Graph()
    mst.add_weighted_edges_from(get_edges_from_matrix(mst_matrix))

    # Plot the original graph
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    pos = nx.spring_layout(graph)
    edge_labels = {(i, j): graph[i][j]['weight'] for i, j in graph.edges()}
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=700, font_size=8, edge_color='gray', width=0.5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Original Graph")

    # Plot the minimum spanning tree
    plt.subplot(122)
    edge_labels_mst = {(i, j): mst[i][j]['weight'] for i, j in mst.edges()}
    nx.draw(mst, pos, with_labels=True, font_weight='bold', node_color='lightcoral', node_size=700, font_size=8, edge_color='black', width=1.5)
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels_mst, font_color='red', font_size=8)
    plt.title("Minimum Spanning Tree")

    plt.show()

def get_edges_from_matrix(matrix):
    edges = []
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if matrix[i, j] != 0:
                edges.append((i, j, matrix[i, j]))
    return edges

def visualize_random_samples(graph_data, mst_answers, num_samples=5):
    # Select random indices for visualization
    indices = np.random.choice(len(graph_data), num_samples, replace=False)

    for idx in indices:
        graph_matrix = graph_data[idx]
        mst_matrix = mst_answers[idx]

        visualize_graph_and_mst(graph_matrix, mst_matrix)

def main():
    # Load the generated graph data and minimum spanning tree answers
    graph_data = joblib.load('graph_data.joblib')
    mst_answers = joblib.load('mst_answers.joblib')

    # Visualize random samples
    visualize_random_samples(graph_data, mst_answers)

    # Convert lists to numpy arrays
    X = np.array(graph_data)
    y = np.array(mst_answers)

    # Define a more complex neural network model with regularization
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1], X.shape[2])),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),  # Added Batch Normalization
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),  # Added Batch Normalization
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(X.shape[1] * X.shape[2], activation='linear'),
        layers.Reshape((X.shape[1], X.shape[2]))
    ])

    # Compile the model with a smaller learning rate
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

    # Implement early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with all available data
    epochs = 50000  # Increase the number of epochs for more training
    batch_size = 32
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    # Save the trained model
    model.save('mst_prediction_model.h5')

if __name__ == "__main__":
    main()
