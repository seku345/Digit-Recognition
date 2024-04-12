import numpy as np


def relu(vector):
    return np.maximum(0, vector)


def softmax(values: np.array) -> np.array:
    exp_values = np.exp(values)

    exp_values_sum = np.sum(exp_values)

    return exp_values / exp_values_sum


def he_initialization(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 /  input_size)


class NeuralNetwork:
    def __init__(self, input_size=784, layer1_size=20, layer2_size=10, out_size=10):

        # weights init
        self.layer1_weights = he_initialization(input_size, layer1_size)
        self.layer2_weights = he_initialization(layer1_size, layer2_size)
        self.out_weights = he_initialization(layer2_size, out_size)

        # biases
        self.layer1_biases = np.zeros(20)
        self.layer2_biases = np.zeros(10)
        self.out_biases = np.zeros(10)

    def feedforward(self, input_vector: np.array):
        h1_result_vector = relu(np.dot(self.layer1_weights, input_vector) + self.layer1_biases)
        h2_result_vector = relu(np.dot(self.layer2_weights, self.layer1_weights) + self.layer2_biases)
        out = np.argmax(np.dot(self.layer2_weights, self.out_weights) + self.out_weights)

        return out

