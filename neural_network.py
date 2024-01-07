from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


class NeuralNetwork:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.classifier = MLPClassifier(solver='lbfgs', random_state=1, max_iter=1000, hidden_layer_sizes=(100, 100))
        self.classifier.fit(self.X_train, self.y_train)

