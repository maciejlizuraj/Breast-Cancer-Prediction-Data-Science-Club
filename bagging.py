from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class Bagging:

    def __init__(self, X_train, X_test, y_train, y_test):
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        self.classifier = self.train_classifier()
        self.evaluate()

    def train_classifier(self):
        parameters = {'n_estimators': [10, 20, 50], 'max_samples': [0.5, 1.0], 'max_features': [0.5, 1.0]}
        bagging_clf = GridSearchCV(BaggingClassifier(), parameters, cv=5)
        bagging_clf.fit(self.X_train_scaled, self.y_train)
        return bagging_clf.best_estimator_

    def evaluate(self):
        y_pred = self.classifier.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("\nBagging accuracy:", accuracy)
        print("Confusion Matrix: ")
        print(confusion_matrix(self.y_test, y_pred))
