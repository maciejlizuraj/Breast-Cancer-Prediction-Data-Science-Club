from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


class NaiveBayes:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.classifier = GaussianNB()
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("\nNaive Bayes accuracy: ", accuracy)
        print("Confusion Matrix: ")
        print(confusion_matrix(y_test, y_pred))
