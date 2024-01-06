import numpy as np
from classification import Classification
import math


# Custom implementation of one-layer neural network with only numpy. Doesn't return great results.

class Network:

    def __init__(self, data_frame, training_set_size):

        self.training_set = data_frame.iloc[0:training_set_size, :]
        self.test_set = data_frame.iloc[training_set_size::, :]
        weights_len = len(self.training_set.axes[1]) - 1

        self.perceptrons = [Perceptron(x, weights_len) for x in Classification]

    def train(self):
        for i in range(len(self.training_set)):
            inputs = self.training_set.iloc[i, 1::].to_numpy()
            classification = self.training_set.iloc[i, 0]

            network_classification = self.classify(inputs)

            for perceptron in self.perceptrons:
                desired_output = 0
                actual_output = 0

                if perceptron.classification == classification:
                    desired_output = 1

                if perceptron.classification == network_classification:
                    actual_output = 1

                perceptron.update_values(desired_output, actual_output, inputs)

        self.training_set = self.training_set.sample(frac=1).reset_index(drop=True)

    def test(self):
        correct = 0
        total = 0
        for i in range(len(self.test_set)):
            real_classification = self.test_set.iloc[i, 0]
            inputs = self.test_set.iloc[i, 1::].to_numpy()

            network_classification = self.classify(inputs)

            total += 1
            if network_classification == real_classification:
                correct += 1
        print('correct:', correct, 'total:', total)

    def classify(self, inputs):

        perceptron_map = {}

        for perceptron in self.perceptrons:
            perceptron_map[perceptron.activate(inputs)] = perceptron.classification

        return perceptron_map[max(perceptron_map.keys())]


class Perceptron:

    def __init__(self, classification, weights_len):
        self.weights = np.random.rand(weights_len, 1)
        self.bias = 0
        self.learning_rate = 0.001
        self.classification = classification

    def activate(self, inputs):
        prod = np.add(np.matmul(inputs, self.weights), self.bias)
        if prod >= 0:
            prod = math.exp(-prod)
            return 1 / (1 + prod)
        else:
            prod = math.exp(prod)
            return prod / (1 + prod)

    def update_values(self, desired_output, actual_output, inputs):
        error = desired_output - actual_output

        d_weights = (inputs * error * self.learning_rate).reshape(4, 1)
        self.weights = np.add(self.weights, d_weights)
        self.bias += self.learning_rate * error * -1
