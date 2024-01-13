from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self, model):
        y_pred = model.predict(self.X_test)
        accuracy = round(accuracy_score(self.y_test, y_pred), 4)
        precision = round(precision_score(self.y_test, y_pred), 4)
        recall = round(recall_score(self.y_test, y_pred), 4)
        f1 = round(f1_score(self.y_test, y_pred), 4)

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
