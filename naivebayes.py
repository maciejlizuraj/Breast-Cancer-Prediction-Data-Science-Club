from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


class NaiveBayes:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.classifier = GaussianNB()
        self.classifier.fit(X_train, y_train)

