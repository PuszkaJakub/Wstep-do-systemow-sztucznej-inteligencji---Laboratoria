import matplotlib.pyplot as plt
import networkx as nx

def visualize_network(network, input_size):
    G = nx.DiGraph()

    input_layer = ['Input {}'.format(i+1) for i in range(input_size)]
    G.add_nodes_from(input_layer, layer='input')

    previous_layer = input_layer

    for l, layer in enumerate(network.layers):
        current_layer = ['Layer {} Neuron {}'.format(l+1, i+1) for i in range(len(layer.neurons))]
        G.add_nodes_from(current_layer, layer='hidden' if l < len(network.layers) - 1 else 'output')

        for prev_node in previous_layer:
            for curr_node in current_layer:
                G.add_edge(prev_node, curr_node)
        
        previous_layer = current_layer

    pos = {}
    layer_nodes = input_layer
    for i, nodes in enumerate([input_layer] + [[f'Layer {l+1} Neuron {n+1}' for n in range(len(layer.neurons))] for l, layer in enumerate(network.layers)]):
        for j, node in enumerate(nodes):
            pos[node] = (i, j - len(nodes) / 2)
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color='yellow', font_size=10, font_weight='bold', arrowsize=20)
    
    input_nodes = [node for node in G.nodes() if 'Input' in node]
    hidden_nodes = [node for node in G.nodes() if 'Layer' in node and 'Neuron' in node and 'Layer 3' not in node]
    output_nodes = [node for node in G.nodes() if 'Layer 3 Neuron' in node]

    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='tomato', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='royalblue', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='limegreen', node_size=3000)

    layer_labels = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer']
    for i, label in enumerate(layer_labels):
        plt.text(i, -3, label, horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    plt.title('Neural Network Visualization')
    plt.show()
