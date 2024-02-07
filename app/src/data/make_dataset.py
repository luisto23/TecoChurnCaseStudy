import pandas as pd
import math
import datetime
from src.utils.config import load_config
from src.utils.store import CaseStudyStore
from src.data.clean_dataset import clean_data

def add_date(idx):
    val = math.floor(idx/541)
    
    u = datetime.datetime.strptime("2021-06-05","%Y-%m-%d")    
    d = datetime.timedelta(days=7*val)
    return u + d

def add_date_df(df):
    df['RecordDate'] = df.index.map(add_date)
    return df


def main():
    store = CaseStudyStore()
    config = load_config()

    raw_df = store.get_raw("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    processed_df = add_date_df(raw_df)
    processed_df = clean_data(processed_df)
    store.put_processed("processed_dataset.csv", processed_df)


if __name__ == "__main__":
    main()