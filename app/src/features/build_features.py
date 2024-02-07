import pandas as pd

from src.features.transformations import (
    transform_catagorical_data,
    transform_numerical_data,
    get_month,
)
from src.utils.store import CaseStudyStore

def main():
    store = CaseStudyStore()

    processed_dataset = store.get_processed("processed_dataset.csv")
    processed_dataset = apply_feature_engineering(processed_dataset)

    store.put_processed("transformed_dataset.csv", processed_dataset)

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(transform_catagorical_data)
        .pipe(transform_numerical_data)
        .pipe(get_month)
    )

if __name__ == "__main__":
    main()