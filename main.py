import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

def read_data():
    data_file = 'data/wpbc.data'

    column_names = ['ID number', 'Outcome', 'Time', 'Nucleus 1 radius', 'Nucleus 1 texture', 'Nucleus 1 perimeter',
                    'Nucleus 1 area', 'Nucleus 1 smoothness', 'Nucleus 1 compactness', 'Nucleus 1 concavity',
                    'Nucleus 1 concave points', 'Nucleus 1 symmetry', 'Nucleus 1 fractal dimension', 'Nucleus 2 radius',
                    'Nucleus 2 texture', 'Nucleus 2 perimeter', 'Nucleus 2 area', 'Nucleus 2 smoothness',
                    'Nucleus 2 compactness', 'Nucleus 2 concavity', 'Nucleus 2 concave points', 'Nucleus 2 symmetry',
                    'Nucleus 2 fractal dimension', 'Nucleus 3 radius', 'Nucleus 3 texture', 'Nucleus 3 perimeter',
                    'Nucleus 3 area', 'Nucleus 3 smoothness', 'Nucleus 3 compactness', 'Nucleus 3 concavity',
                    'Nucleus 3 concave points', 'Nucleus 3 symmetry', 'Nucleus 3 fractal dimension', 'Tumor size',
                    'Lymph node status']
    df = pd.read_csv(data_file, names=column_names)
    df.drop('ID number', axis=1, inplace=True)
    df.drop(df.loc[df['Lymph node status'] == '?'].index, inplace=True)

    return df


def bayes(df):
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)
    bayes_accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", bayes_accuracy)
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))


bayes(read_data())



