# data_preparation.py
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from environment import SEED


def load_data(file_path):
    df = pd.read_csv(file_path)
    df["booking status"].replace({"Canceled": 1, "Not_Canceled": 0}, inplace=True)  # noqa
    return df


def split_and_save_data(
    data, train_A_path, train_B_path, test_path, test_size=0.2, split_ratio=0.5
):
    train_val_set, test_set = train_test_split(
        data, test_size=test_size, random_state=SEED
    )
    test_set.to_csv(test_path, index=False)
    print(f'{test_path} file created...')

    train_set_A, train_set_B = train_test_split(
        train_val_set, test_size=split_ratio, random_state=SEED
    )
    train_set_A.to_csv(train_A_path, index=False)
    print(f'{train_A_path} file created...')
    train_set_B.to_csv(train_B_path, index=False)
    print(f'{train_B_path} file created...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Preparation for Hotel Booking Cancellation Prediction"  # noqa
    )
    parser.add_argument(
        "input_file", type=str, help="Input CSV file containing the booking data"  # noqa
    )
    parser.add_argument(
        "train_A_file", type=str, help="Output CSV file path for training set A"  # noqa
    )
    parser.add_argument(
        "train_B_file", type=str, help="Output CSV file path for training set B"  # noqa
    )
    parser.add_argument(
        "test_file", type=str, help="Output CSV file path for the test set"  # noqa
    )
    args = parser.parse_args()

    data = load_data(args.input_file)
    split_and_save_data(data, args.train_A_file, args.train_B_file, args.test_file)  # noqa
    print('Completed!')
