from sklearn.model_selection import GridSearchCV
from sklearn import svm


class SupportVectorMachine:
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.classifier = svm.SVC(C=10, gamma=0.1, kernel='rbf', random_state=42)
        self.classifier.fit(X_train, y_train)
        # self.grid_search()

    def get_classifier(self):
        return self.classifier

    def grid_search(self):
        parameters = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        }

        grid_search = GridSearchCV(self.classifier, parameters, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best hyperparameters:", grid_search.best_params_)
