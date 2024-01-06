import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
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


def data_preprocessing(df):
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def no_processing(df):
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test


def bayes(df):
    #X_train, X_test, y_train, y_test = no_processing(df)
    X_train, X_test, y_train, y_test = data_preprocessing(df)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))


bayes(read_data())

