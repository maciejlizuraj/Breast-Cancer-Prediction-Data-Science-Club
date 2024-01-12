from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost:

    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.classifier = AdaBoostClassifier(n_estimators=300, learning_rate=1.0, random_state=42)
        self.classifier.fit(X_train, y_train)
        # self.grid_search()

    def get_classifier(self):
        return self.classifier

    def grid_search(self):
        parameters = {
            'n_estimators': [200, 300, 400],
            'learning_rate': [1.0, 1.5, 2.0],
        }

        grid_search = GridSearchCV(self.classifier, parameters, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best hyperparameters:", grid_search.best_params_)
