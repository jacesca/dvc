from typing import List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ExampleML2.utils_and_constants import load_config
import argparse


def read_dataset(
    filename: str, drop_columns: List[str], target_column: str
) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe
    Target column values are expected in binary format with Yes/No values

    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    df = pd.read_csv(filename).drop(columns=drop_columns)
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})
    return df


def target_encode_categorical_features(
    df: pd.DataFrame, categorical_columns: List[str], target_column: str
) -> pd.DataFrame:
    """
    Target encodes the categorical features of the dataframe
    (http://www.saedsayad.com/encoding.htm)


    Parameters:
    df (pd.Dataframe): Pandas dataframe containing features and targets
    categorical_columns (List[str]): categorical column names that will be
    target encoded target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    encoded_data = df.copy()

    # Iterate through categorical columns
    for col in categorical_columns:
        # Calculate mean target value for each category
        encoding_map = df.groupby(col)[target_column].mean().to_dict()

        # Apply target encoding
        encoded_data[col] = encoded_data[col].map(encoding_map)

    return encoded_data


def impute_and_scale_data(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes numerical data to its mean value
    and then scales the data to a normal distribution

    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Imputed and Scaled dataframe
    """

    # Impute data with mean strategy
    imputer = SimpleImputer(strategy="mean")
    X_preprocessed = imputer.fit_transform(df_features.values)

    # Scale and fit with zero mean and unit variance
    scaler = StandardScaler()
    X_preprocessed = scaler.fit_transform(X_preprocessed)

    return pd.DataFrame(X_preprocessed, columns=df_features.columns)


def main(config_file: str, raw_dataset: str, processed_dataset: str):
    config = load_config(config_file)["preprocess"]
    drop_colnames = config["drop_colnames"]
    target_column = config["target_column"]

    # Read dataset
    print("Reading raw data and processing it...")
    weather = read_dataset(
        filename=raw_dataset, drop_columns=drop_colnames,
        target_column=target_column
    )

    # Target encode categorical columns
    # results in all columns becoming numerical
    print("Target encoding categorical columns...")
    categorical_columns = config["categorical_features"]
    weather = target_encode_categorical_features(
        df=weather, categorical_columns=categorical_columns,
        target_column=target_column
    )

    # Impute and scale features
    print("Imputing and scaling features...")
    weather_features_processed = impute_and_scale_data(
        weather.drop(columns=target_column, axis=1)
    )

    # Write processed dataset
    print(f"Writing processed dataset to {processed_dataset}...")
    weather_labels = weather[target_column]
    weather = pd.concat([weather_features_processed, weather_labels], axis=1)
    weather.to_csv(processed_dataset, index=None)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess data for the weather prediction model"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Configuration file with parameters",
        default="config.yaml",
    )
    parser.add_argument(
        "input_dataset",
        type=str,
        help="CSV file with the raw dataset",
        default="raw_dataset/weather.csv",
    )
    parser.add_argument(
        "output_dataset",
        type=str,
        help="Processed CSV file path",
        default="processed_dataset/weather.csv",
    )
    args = parser.parse_args()
    main(
        config_file=args.config_file,
        raw_dataset=args.input_dataset,
        processed_dataset=args.output_dataset,
    )
