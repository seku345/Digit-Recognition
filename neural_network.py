import numpy as np


def relu(vector):
    return np.maximum(0, vector)


def softmax(values: np.array) -> np.array:
    shift = values - np.max(values)

    exp_values = np.exp(shift)

    exp_values_sum = np.sum(exp_values)

    return exp_values / exp_values_sum


def he_initialization(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 /  input_size)


class NeuralNetwork:

    LEARN_RATE = 0.1
    EPOCHS = 1000

    def __init__(self, input_size=784, layer1_size=20, layer2_size=10, out_size=10) -> None:

        # weights init
        self.W1 = he_initialization(input_size, layer1_size)
        self.W2 = he_initialization(layer1_size, layer2_size)
        self.W3 = he_initialization(layer2_size, out_size)

        # biases
        self.B1 = np.zeros(20)
        self.B2 = np.zeros(10)
        self.B3 = np.zeros(10)

    def feedforward(self, input_vector: np.array) -> np.array:
        h1_result_vector = relu(np.dot(self.W1, input_vector) + self.B1)
        h2_result_vector = relu(np.dot(self.W2, h1_result_vector) + self.B2)
        out = softmax(np.dot(self.W3, h2_result_vector) + self.B3)

        return out

    def train(self, data, answers):
        for epoch in range(self.EPOCHS):
            for X, answer in zip(data, answers):

                # feedforward
                h1 = relu(np.dot(self.W1, X) + self.B1)
                h2 = relu(np.dot(self.W2, h1) + self.B2)
                pred = softmax(np.dot(self.W3, h2) + self.B3)

                # back propagation
                d_out = pred - answer

                d_W3 = np.dot(d_out, h2.T)
                d_b3 = np.sum(d_out, axis=1, keepdims=True)

                d_h2 = np.dot(self.W3.T, d_out)
                d_h2[h2 <= 0] = 0

                d_W2 = np.dot(d_h2, h1.T)
                d_b2 = np.sum(d_h2, axis=1, keepdims=True)

                d_h1 = np.dot(self.W2.T, d_h2)
                d_h1[h1 <= 0] = 0

                d_W1 = np.dot(d_h1, X.T)
                d_b1 = np.sum(d_h1, axis=1, keepdims=True)

                # updating weights and biases
                self.W1 -= self.LEARN_RATE * d_W1
                self.B1 -= self.LEARN_RATE * d_b1
                self.W2 -= self.LEARN_RATE * d_W2
                self.B2 -= self.LEARN_RATE * d_b2
                self.W3 -= self.LEARN_RATE * d_W3
                self.B3 -= self.LEARN_RATE * d_b3

            if epoch % 10 == 0:
                predictions = np.apply_along_axis(self.feedforward, 1, data)
                loss = cross_entropy(answers, predictions)
                print(f'Epoch {epoch} loss: {loss:.3f}')


def cross_entropy(answers: np.array, predictions: np.array) -> float:
    n = answers.shape[0]

    loss = -1 / n * np.sum(answers * np.log(predictions))

    return loss


if __name__ == '__main__':
    ...
