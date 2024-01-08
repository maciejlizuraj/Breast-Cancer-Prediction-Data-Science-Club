from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


class AdaBoost:

    def __init__(self, X_train, y_train):
        parameters = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]}
        self.classifier = GridSearchCV(AdaBoostClassifier(), parameters, cv=5)
        self.classifier.fit(X_train, y_train)

    def get_classifier(self):
        return self.classifier
