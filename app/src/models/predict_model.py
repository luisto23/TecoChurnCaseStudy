import numpy as np
import pandas as pd

from src.features.build_features import apply_feature_engineering
from src.utils.guardrails import validate_prediction_results
from src.utils.store import CaseStudyStore


@validate_prediction_results
def main():

    store = CaseStudyStore()

    df_test = store.get_raw("test_data.csv")
    print('read_dome')
    df_test = apply_feature_engineering(df_test)

    model = store.get_model("saved_model.pkl")
    df_test["Churn"] = model.predict(df_test)
    print('test')
    results = df_test[['customerID', 'Churn']]
    store.put_predictions("results.csv", results)

if __name__ == '__main__':
    main()