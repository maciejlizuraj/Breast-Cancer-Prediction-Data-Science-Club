from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class NeuralNetwork:

    def __init__(self, df):
        df['Classification'] = df['Classification'].astype(str)

        X = df.drop('Classification', axis=1)
        y = df['Classification']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        self.X_train = preprocessing.StandardScaler().fit(self.X_train).transform(self.X_train)
        self.X_test = preprocessing.StandardScaler().fit(self.X_test).transform(self.X_test)
        self.classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
        self.classifier = MLPClassifier(solver='lbfgs', random_state=1, max_iter=1000, hidden_layer_sizes=(3, 1))

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)
        self.classifier.predict(self.X_test)

    def test(self):
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
