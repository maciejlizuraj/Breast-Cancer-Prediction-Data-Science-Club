from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, X_train, y_train):
        self.classifier = GaussianNB()
        self.classifier.fit(X_train, y_train)

    def get_classifier(self):
        return self.classifier
