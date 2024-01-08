from sklearn import svm


class SupportVectorMachine:
    def __init__(self, X_train, y_train):
        self.classifier = svm.SVC()
        self.classifier.fit(X_train, y_train)

    def get_classifier(self):
        return self.classifier

