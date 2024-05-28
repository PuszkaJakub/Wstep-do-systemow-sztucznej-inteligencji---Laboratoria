import numpy as np
from network import Network
from layer import Layer
from visualization import visualize_network

network = Network()

input_size = 3  
layer_1_size = 4 
layer_2_size = 4  
output_size = 1  

network.add_layer(Layer(layer_1_size, input_size))
network.add_layer(Layer(layer_2_size, layer_1_size))
network.add_layer(Layer(output_size, layer_2_size))

input_data = np.array([0.1, 0.78, 0.116])

output = network.forward(input_data)
print("Output:", output)

visualize_network(network, input_size)
