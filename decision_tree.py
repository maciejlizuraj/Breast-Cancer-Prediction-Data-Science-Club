from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, X_train, y_train):
        parameters = {'max_depth': [5, 10, 15], 'min_samples_leaf': [2, 4, 6]}
        self.classifier = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5)
        self.classifier.fit(X_train, y_train)

    def get_classifier(self):
        return self.classifier
