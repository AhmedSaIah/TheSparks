import pandas as pd

def load_data():
    url = 'https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view'
    file_id = url.split('/')[5]
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    try:
        df = pd.read_csv(download_url)
        print("Data loaded successfully:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None