import pickle
import numpy as np
import pandas as pd


def relu(vector):
    return np.maximum(0, vector)


def softmax(values: np.array) -> np.array:
    shift = values - np.max(values)

    exp_values = np.exp(shift)

    exp_values_sum = np.sum(exp_values)

    return exp_values / exp_values_sum


def he_initialization(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)


class NeuralNetwork:

    def __init__(self, trained=False, learning_rate=0.05, epochs=1000, input_size=784, layer1_size=20, layer2_size=10,
                 out_size=10) -> None:

        self.LEARN_RATE = learning_rate
        self.EPOCHS = epochs
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size

        if not trained:
            # weights init
            self.W1 = he_initialization(layer1_size, input_size)
            self.W2 = he_initialization(layer2_size, layer1_size)
            self.W3 = he_initialization(out_size, layer2_size)

            # biases
            self.B1 = np.zeros(layer1_size)
            self.B2 = np.zeros(layer2_size)
            self.B3 = np.zeros(out_size)

        else:
            with (open('weights.pkl', 'rb') as file):
                (self.LEARN_RATE, self.EPOCHS, self.layer1_size, self.layer2_size,
                 self.W1, self.B1, self.W2, self.B2, self.W3, self.B3) = pickle.load(file)

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

                d_W3 = np.dot(d_out[:, None], h2[None, :])
                d_b3 = np.sum(d_out, axis=0, keepdims=True)

                d_h2 = np.dot(self.W3.T, d_out)
                d_h2[h2 <= 0] = 0

                d_W2 = np.dot(d_h2[:, None], h1[None, :])
                d_b2 = np.sum(d_h2, axis=0, keepdims=True)

                d_h1 = np.dot(self.W2.T, d_h2)
                d_h1[h1 <= 0] = 0

                d_W1 = np.dot(d_h1[:, None], X[None, :])
                d_b1 = np.sum(d_h1, axis=0, keepdims=True)

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

        self.save_model()

    # saving weights and biases
    def save_model(self):
        with open('weights.pkl', 'wb') as file:
            pickle.dump([self.LEARN_RATE, self.EPOCHS, self.layer1_size, self.layer2_size,
                         self.W1, self.B1, self.W2, self.B2, self.W3, self.B3], file)


def cross_entropy(answers: np.array, predictions: np.array) -> float:
    n = answers.shape[0]

    loss = -1 / n * np.sum(answers * np.log(predictions + 1e-8))

    return loss


def find_the_best_model():
    learn_rate_range = [0.001, 0.005, 0.01, 0.05, 0.1]
    epochs_range = [250, 500, 750, 1000, 1250, 1500]
    layer1_size_range = [12, 16, 20, 24, 28]
    layer2_size_range = [5, 10, 15]

    best_model = None
    best_accuracy = 0

    for _ in range(2):
        learn_rate = np.random.choice(learn_rate_range)
        epochs = np.random.choice(epochs_range)
        layer1_size = np.random.choice(layer1_size_range)
        layer2_size = np.random.choice(layer2_size_range)

        model = NeuralNetwork(learning_rate=learn_rate, epochs=epochs, layer1_size=layer1_size, layer2_size=layer2_size)
        model.train(inputs_train, labels_one_hot_train)

        predictions_test = np.apply_along_axis(model.feedforward, 1, inputs_test)
        predicted_labels_test = np.argmax(predictions_test, axis=1)
        accuracy = np.sum(predicted_labels_test == labels_test) / len(labels_test)

        print(f'Current accuracy: {accuracy}')
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

    print(f'Best accuracy: {best_accuracy}')
    best_model.save_model()


if __name__ == '__main__':
    # reading training data
    df_train = pd.read_csv('data/mnist_train.csv')

    data_train = df_train.values

    labels_train = data_train[:, 0]
    inputs_train = data_train[:, 1:]

    inputs_train = (inputs_train > 0).astype(int)

    labels_one_hot_train = np.eye(10)[labels_train.astype(int)]

    # training
    # network = NeuralNetwork(trained=False)
    # network.train(inputs_train, labels_one_hot_train)

    # reading testing data
    df_test = pd.read_csv('data/mnist_test.csv')

    data_test = df_test.values

    labels_test = data_test[:, 0]
    inputs_test = data_test[:, 1:]

    inputs_test = (inputs_test > 0).astype(int)

    labels_one_hot_test = np.eye(10)[labels_test.astype(int)]

    # testing
    # network_trained = NeuralNetwork(trained=True)

    find_the_best_model()

    network = NeuralNetwork(trained=True)
    predictions_test = np.apply_along_axis(network.feedforward, 1, inputs_test)

    accuracy = np.sum(np.argmax(predictions_test, axis=1) == labels_test) / len(labels_test)

    print(f'Total accuracy: {accuracy:.3f}')
