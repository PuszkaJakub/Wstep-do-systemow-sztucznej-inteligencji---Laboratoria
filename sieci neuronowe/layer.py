from neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, n_neurons, n_inputs_per_neuron):
        self.neurons = [Neuron(n_inputs_per_neuron) for _ in range(n_neurons)]

    def __call__(self, inputs):
        return np.array([neuron(inputs) for neuron in self.neurons])