from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, X_train, y_train):
        self.classifier = MLPClassifier(solver='lbfgs', random_state=1, max_iter=1000, hidden_layer_sizes=(100, 100))
        self.classifier.fit(X_train, y_train)

    def get_classifier(self):
        return self.classifier
