import os
import pandas as pd

def preprocessing(df):  # this function cleans the dataset
    
    n_before = df.shape[0]  # Stores the number of rows before cleaning

    df = df.dropna()  # Removes rows with missing values
    df = df.drop_duplicates(subset=["timestamp"])  # Removes duplicate timestamps

    n_after = df.shape[0]
    print(f"Size before: {n_before} | Size after: {n_after}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%dT%H:%M"  # Converts the timestamp column to datetime format
    )

    df = df.set_index("timestamp")  # Uses the timestamp as the dataframe index

    return df


def read_dataset(name):
    path = os.getcwd()

    if name == "O3":
        file_path = os.path.join(path, "Ref-Data", "O3_all.csv")

    elif name == "NO2":
        file_path = os.path.join(path, "Ref-Data", "NO2_all.csv")

    elif name == "NO2_O3":
        file_path = os.path.join(path, "Ref-Data", "NO2_O3_all.csv")

    else:
        raise ValueError(
            "Dataset name must be: O3, NO2, or NO2_O3"
        )

    df = pd.read_csv(file_path, sep=";", header=0)

    return df
