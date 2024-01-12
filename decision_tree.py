from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.classifier = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=2, splitter='random', random_state=42)
        self.classifier.fit(X_train, y_train)
        # self.grid_search()

    def grid_search(self):
        parameters = {'max_depth': [10, 20, 30, 50], 'min_samples_leaf': [2, 4, 6, 8, 10], 'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'], }

        grid_search = GridSearchCV(self.classifier, parameters, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best hyperparameters:", grid_search.best_params_)

    def get_classifier(self):
        return self.classifier
