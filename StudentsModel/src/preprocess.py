import pandas as pd

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(filepath, header=None, names=['Hours', 'Scores'])
    return df
