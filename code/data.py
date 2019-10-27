import pandas as pd
import numpy as np


def load_data_with_gender(
    new_data_path="../data/Discharge destinations including Gender.xlsx",
    old_data_path="../data/Discharge destinations 2.xlsx",
):

    """
    Load the data into a dataframe.

    Because the data including Gender does not have the same destination mapping as the original data, we read in
    the original, map using the supplied outcome mapping, and confirm the resulting data is identical.
    """
    original_df = pd.read_excel(
        old_data_path, sheet_name="Regression analysis", header=2
    )
    original_df.reset_index(inplace=True)
    original_df.drop(["index", "Key", "Unnamed: 0"], axis=1, inplace=True)

    outcome_dict = {0: "Death", 1: "Inpatient", 2: "CH", 3: "Home"}
    original_df["Outcome"] = original_df["Outcome"].map(lambda x: outcome_dict[x])

    new_df = pd.read_excel(
        new_data_path,
        sheet_name="Gender",
        header=1,
        usecols=[0, 1, 2, 3, 5],
        colnames=["Age", "NIHSS", "MRS", "New Destination", "Gender"],
    )

    new_df.rename({"Unnamed: 3": "New Outcome"}, inplace=True, axis=1)

    NEW_TO_OLD_OUTCOME_MAPPING = {
        "T": "Inpatient",
        "TC": "CH",
        "D": "Death",
        "H": "Home",
        "TN": "Inpatient",
        "TCN": "CH",
    }

    new_df["Outcome"] = new_df["New Outcome"].map(NEW_TO_OLD_OUTCOME_MAPPING)

    assert np.array_equal(
        new_df["Outcome"].value_counts().values,
        original_df["Outcome"].value_counts().values,
    )
    assert np.array_equal(
        new_df.groupby("Outcome").sum().values,
        original_df.groupby("Outcome").sum().values,
    )

    new_df.drop("New Outcome", axis=1, inplace=True)

    return new_df


def stratified_sample_df(df, col, frac, random_state=1234):
    if frac > 1.0:
        raise ValueError

    original_numbers = df[col].value_counts()
    sample_n = {k: int(v * frac) for k, v in original_numbers.items()}

    df_ = df.groupby(col).apply(
        lambda x: x.sample(sample_n[x.name], random_state=random_state)
    )
    df_.index = df_.index.droplevel(0)

    return df_
