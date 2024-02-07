import pandas as pd

def clean_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
    df = df.dropna()
    return df