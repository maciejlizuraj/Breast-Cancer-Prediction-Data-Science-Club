from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class Bagging:

    def __init__(self, X_train, y_train):
        parameters = {'n_estimators': [10, 20, 50], 'max_samples': [0.5, 1.0], 'max_features': [0.5, 1.0]}
        self.classifier = GridSearchCV(BaggingClassifier(), parameters, cv=5)
        self.classifier.fit(X_train, y_train)

    def get_classifier(self):
        return self.classifier