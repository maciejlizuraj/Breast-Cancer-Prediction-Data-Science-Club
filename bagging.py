from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier


class Bagging:

    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.classifier = BaggingClassifier()
        self.classifier.fit(X_train, y_train)
        # self.grid_search()

    def get_classifier(self):
        return self.classifier

    def grid_search(self):
        parameters = {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.2, 0.5, 1.0],
            'max_features': [0.5, 1.0, 1.5],
        }

        grid_search = GridSearchCV(self.classifier, parameters, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best hyperparameters:", grid_search.best_params_)
