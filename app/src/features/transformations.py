import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def transform_catagorical_data(df: pd.DataFrame) -> pd.DataFrame:
    labelencoder = LabelEncoder()
    df['gender'] = labelencoder.fit_transform(df['gender'])
    df['Partner'] = labelencoder.fit_transform(df['Partner'])
    df['Dependents'] = labelencoder.fit_transform(df['Dependents'])
    df['PhoneService'] = labelencoder.fit_transform(df['PhoneService'])
    df['MultipleLines'] = labelencoder.fit_transform(df['MultipleLines'])
    df['InternetService'] = labelencoder.fit_transform(df['InternetService'])
    df['OnlineSecurity'] = labelencoder.fit_transform(df['OnlineSecurity'])
    df['DeviceProtection'] = labelencoder.fit_transform(df['DeviceProtection'])
    df['TechSupport'] = labelencoder.fit_transform(df['TechSupport'])
    df['StreamingTV'] = labelencoder.fit_transform(df['StreamingTV'])
    df['StreamingMovies'] = labelencoder.fit_transform(df['StreamingMovies'])
    df['Contract'] = labelencoder.fit_transform(df['Contract'])
    df['PaperlessBilling'] = labelencoder.fit_transform(df['PaperlessBilling'])
    df['PaymentMethod'] = labelencoder.fit_transform(df['PaymentMethod'])
    df['Churn'] = labelencoder.fit_transform(df['Churn'])
    return df

def transform_numerical_data(df: pd.DataFrame) -> pd.DataFrame:
    scalar = MinMaxScaler()
    df['tenure'] = scalar.fit_transform(df[['tenure']])
    df['MonthlyCharges'] = scalar.fit_transform(df[['MonthlyCharges']])
    df['TotalCharges'] = scalar.fit_transform(df[['TotalCharges']])
    return df
    
def get_month(df: pd.DataFrame) -> pd.DataFrame:
    df['RecordDate'] = pd.to_datetime(df['RecordDate'])
    df['month'] = df['RecordDate'].dt.month
    return df
