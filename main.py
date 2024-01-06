import pandas as pd
from classification import Classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def label_classification(row):
    if row['Outcome'] == 'R' and row['Time'] < 24:
        return Classification.POSITIVE
    if row['Outcome'] == 'N' and row['Time'] > 24:
        return Classification.NEGATIVE
    return Classification.NEITHER


def label_mean_texture(row):
    return (row['Nucleus 1 texture'] + row['Nucleus 2 texture'] + row['Nucleus 3 texture']) / 3


def label_worst_area(row):
    return max(row['Nucleus 1 area'], row['Nucleus 2 area'], row['Nucleus 3 area'])


def label_worst_concavity(row):
    return max(row['Nucleus 1 concave points'], row['Nucleus 2 concave points'], row['Nucleus 3 concave points'])


def label_worst_fractal_dimension(row):
    return max(row['Nucleus 1 fractal dimension'], row['Nucleus 2 fractal dimension'],
               row['Nucleus 3 fractal dimension'])


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
    df.drop(df.loc[df['Lymph node status'] == '?'].index, inplace=True)
    df['Lymph node status'] = df['Lymph node status'].astype(float)
    df['Classification'] = df.apply(label_classification, axis=1)
    df['Mean texture'] = df.apply(label_mean_texture, axis=1)
    df['Mean area'] = df.apply(label_worst_area, axis=1)
    df['Mean concavity'] = df.apply(label_worst_concavity, axis=1)
    df['Mean dimension'] = df.apply(label_worst_fractal_dimension, axis=1)
    df = df.filter(['Classification', 'Mean texture', 'Mean area', 'Mean concavity', 'Mean dimension'])

    return df


def naive_bayes_classifier(df):
    # Casting 'Classification' into string, because sklearn can't use enum
    df['Classification'] = df['Classification'].astype(str)

    X = df.drop('Classification', axis=1)
    y = df['Classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    data_frame = read_data()
    print(data_frame)
    # here add function/initializers that take data_frame as an argument
    naive_bayes_classifier(data_frame)

