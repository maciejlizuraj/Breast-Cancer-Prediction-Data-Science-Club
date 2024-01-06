from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class DecisionTree:

    def __init__(self, X_train, X_test, y_train, y_test):
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        self.y_train = y_train
        self.y_test = y_test
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.classifier = self.train_classifier()
        self.evaluate()

    def train_classifier(self):
        parameters = {'max_depth': [5, 10, 15], 'min_samples_leaf': [2, 4, 6]}
        dt_clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5)
        dt_clf.fit(self.X_train_scaled, self.y_train)
        return dt_clf.best_estimator_

    def evaluate(self):
        y_pred = self.classifier.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("\nDecision Tree Accuracy:", accuracy)
        print("Confusion Matrix: ")
        print(confusion_matrix(self.y_test, y_pred))
