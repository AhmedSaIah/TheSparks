import pandas as pd

def encode_species(df):
    df_encoded = pd.get_dummies(df, columns=['Species'], prefix='Species')
    df_encoded.drop(columns=['Id'], inplace=True)
    return df_encoded

def split_features_labels(df_encoded):
    X = df_encoded.drop(columns=[col for col in df_encoded.columns if 'Species_' in col])
    y = df_encoded[[col for col in df_encoded.columns if 'Species_' in col]]
    return X, y

def label_encoding(df):
    y_labels = df['Species'].replace({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    return y_labels
