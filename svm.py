from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score


class SupportVectorMachine:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.clf = svm.SVC()
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("\nSVM accuracy:", accuracy)
        print("Confusion Matrix: ")
        print(confusion_matrix(y_test, y_pred))
