from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.classifier = MLPClassifier(solver='lbfgs', random_state=42, max_iter=100, alpha=0.01, hidden_layer_sizes=(30, 30))
        self.classifier.fit(X_train, y_train)
        # self.grid_search()

    def get_classifier(self):
        return self.classifier

    def grid_search(self):
        parameters = {
            'hidden_layer_sizes': [(10, 10), (20, 20), (30, 30)],
            'max_iter': [10, 50, 100],
            'alpha': [0.001, 0.01, 0.1],
        }

        grid_search = GridSearchCV(self.classifier, parameters, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best hyperparameters:", grid_search.best_params_)
