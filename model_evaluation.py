from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def display_evaluation(self, model):
        y_pred = model.predict(self.X_test)
        accuracy = round(accuracy_score(self.y_test, y_pred), 4)
        precision = round(precision_score(self.y_test, y_pred), 4)
        recall = round(recall_score(self.y_test, y_pred), 4)
        f1 = round(f1_score(self.y_test, y_pred), 4)

        print('Evaluation Metrics:')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print()
