from sklearn.metrics import accuracy_score, confusion_matrix


class ModelEvaluator:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None

    def evaluate(self, model):
        self.y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, self.y_pred)

        print("Accuracy:", accuracy)

        cm = confusion_matrix(self.y_test, self.y_pred)
        print("Confusion Matrix:")
        print(cm)