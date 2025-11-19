import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning and preprocessing:
    - Handle missing values
    - Convert last_review to datetime
    - Create days_since_last_review
    - Create one-hot encoded room_type
    - Drop unused columns
    """
    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Create days_since_last_review
    df["days_since_last_review"] = (
        pd.Timestamp("today") - df["last_review"]
    ).dt.days

    df["days_since_last_review"] = df["days_since_last_review"].fillna(
        df["days_since_last_review"].median()
    )

    # One-hot encode room_type only
    df = pd.get_dummies(df, columns=["room_type"], drop_first=True)

    # Drop unused columns
    df = df.drop(
        columns=[
            "id",
            "name",
            "host_name",
            "last_review",
            "neighbourhood",
        ],
        errors="ignore",
    )

    return df


def save_processed(df: pd.DataFrame, output_path: str):
    """Saves the cleaned dataframe to CSV."""
    df.to_csv(output_path, index=False)
