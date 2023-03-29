import itertools
import json
import typing
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd

from . import json_parsing


def count_ones_sets(combination: list | tuple) -> int:
    """Count the number of lists containing sequence of one

    Args:
        combination (list | tuple): the chromosomes combination

    Returns:
        int: the number of list of ones

    Example:
        count_ones_sets([1, 0, 1, 0, 1, 0, 1, 1, 0]) -> count = 4
    """

    count = 0
    for i, j in zip(combination, combination[1:] + [0]):
        if i == 1 and j == 0:
            count += 1

    return count


def read_json_file(file_path):
    with open(file_path, mode="r") as f:
        file = json.load(f, object_hook=json_parsing.decode_date_time)
    return file


def save_to_json(file_path):
    def inner(func: Callable[[str], dict]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            spaces_combs_details = func(*args, **kwargs)

            # Save to json file
            with open(file_path, mode="w") as f:
                json.dump(spaces_combs_details, f, indent=4, cls=json_parsing.DateTimeEncoder)

            return spaces_combs_details

        return wrapper

    return inner


class SpaceDetailsParser:
    def __init__(self, num_enc_bits) -> None:
        self.num_enc_bits = num_enc_bits

    def __normalize_json(self, data: dict[dict["str" : typing.Any]]) -> pd.DataFrame:
        """Normalize the json data for further preprocessing

        Args:
            data (dict): available reservations of spaces

        Returns:
            pd.DataFrame: dataframe of normalized spaces
        """
        spaces_details = pd.json_normalize(data, record_path=["available_dates"], meta=["id"], errors="ignore")

        # Change columns type
        spaces_details["start"] = spaces_details["start"].astype("datetime64[ns]")
        spaces_details["end"] = spaces_details["end"].astype("datetime64[ns]")
        spaces_details["cancellable"] = spaces_details["cancellable"].astype(int)

        # Split date from time
        spaces_details["date"] = spaces_details["start"].dt.to_period("d").astype("datetime64[ns]")
        # spaces_details["date"] = spaces_details["date"].astype("str")

        return spaces_details

    def __encode_time_span(self, spaces_details: pd.DataFrame) -> pd.DataFrame:
        """Encode the time span of each space for each available time

        Args:
            spaces_details (pd.DataFrame): spaces details

        Returns:
            pd.DataFrame: spaces details with new encoded columns
        """

        def encode(row: pd.Series):
            if row["end"].hour == 23:
                end = str(row["end"].hour)
            else:
                end = str(row["end"].hour - 1)
            start = str(row["start"].hour)

            row.loc[start:end] = 1

            return row

        # Create matrix of zeros with shaped number of available spaces' time span
        # and num of intended encoded bits
        zeros_matrix = np.zeros(shape=(spaces_details.shape[0], self.num_enc_bits), dtype="int")

        zeros_matrix = pd.DataFrame(zeros_matrix, columns=[str(i) for i in range(self.num_enc_bits)])

        spaces_details = pd.concat([spaces_details, zeros_matrix], axis=1)

        spaces_details = spaces_details.apply(encode, axis=1)

        return spaces_details

    def __add_total_time_span(self, spaces_details: pd.DataFrame) -> pd.DataFrame:
        spaces_details["total_time_span"] = spaces_details.loc[:, "0":"23"].sum(axis=1)

        return spaces_details

    def get_data(self, data: dict[dict["str" : typing.Any]]) -> dict:
        """Restructure the spaces details

        Returns:
            dict: spaces data
        """

        spaces_details = self.__normalize_json(data)
        spaces_details = self.__encode_time_span(spaces_details)
        spaces_details = self.__add_total_time_span(spaces_details)

        return spaces_details


class SpacesCombinationsParser:
    def __create_space_combinations(self, spaces_details: pd.DataFrame) -> list:
        """Create combinations from the available spaces

        Args:
            spaces_details (pd.DataFrame): available spaces

        Returns:
            list: list of created combinations
        """

        unique_spaces = spaces_details["id"].unique().tolist()

        combinations_lst = list(
            itertools.chain.from_iterable(
                itertools.combinations(unique_spaces, r) for r in range(2, len(unique_spaces) + 1)
            )
        )

        return combinations_lst

    def __get_combinations_details(self, spaces_details: pd.DataFrame, combinations: list[str]) -> pd.DataFrame:
        """Get the related data for the created combinations

        Args:
            spaces_details (pd.DataFrame): spaces details
            combinations (list[str]): created combinations

        Returns:
            pd.DataFrame: new spaces details containing created combinations
        """

        spaces_combs_details = spaces_details.groupby(["id", "date"]).sum(numeric_only=True)

        for comb in combinations:
            df_comb = pd.DataFrame(spaces_details.query("id in @comb").groupby("date").sum(numeric_only=True))

            # Edit only 0->23 columns to be 1
            df_comb.loc[:, "0":"23"][(df_comb.loc[:, "0":"23"] > 1)] = 1

            df_comb = pd.concat({"".join(comb): df_comb}, names=["id"])

            spaces_combs_details = pd.concat([spaces_combs_details, df_comb], axis=0)

        spaces_combs_details[(spaces_combs_details > 1)] = 1

        # Recalculate total time span
        spaces_combs_details["total_time_span"] = spaces_combs_details.loc[:, "0":"23"].sum(axis=1)

        # Recalculate cancellable span
        spaces_combs_details["cancellable_span"] = (
            spaces_combs_details["total_time_span"] * spaces_combs_details["cancellable"]
        )

        return spaces_combs_details

    def __add_cancellable_percent(self, spaces_combs_details):
        spaces_combs_details = spaces_combs_details.groupby("id")[["total_time_span", "cancellable_span"]].sum()
        spaces_combs_details["cancellable_percent"] = (
            spaces_combs_details["cancellable_span"] / spaces_combs_details["cancellable_span"].sum()
        )

        return spaces_combs_details

    def get_data(self, spaces_data: pd.DataFrame) -> pd.DataFrame:
        combinations = self.__create_space_combinations(spaces_data)
        spaces_combs_details = self.__get_combinations_details(spaces_data, combinations)
        spaces_combs_details = self.__add_cancellable_percent(spaces_combs_details)
        return spaces_combs_details


class DecisionMatrix:
    def __init__(self, goal_time) -> None:
        self.goal_time = goal_time

    def __add_num_spaces_var(self, spaces_details: pd.DataFrame) -> pd.DataFrame:
        spaces_details["num_spaces"] = spaces_details.index.str.split(pat=r"\d", regex=True)
        spaces_details["num_spaces"] = spaces_details["num_spaces"].apply(len) - 1

        return spaces_details

    def __add_distance_var(self, spaces_details: pd.DataFrame, goal_time: int) -> pd.DataFrame:
        spaces_details["distance"] = (spaces_details["total_time_span"] - goal_time).abs()
        spaces_details.drop(columns=["total_time_span"], inplace=True)
        return spaces_details

    def create(self, spaces_details: pd.DataFrame) -> pd.DataFrame:
        """Create decision matrix for ranking algorithm

        Args:
            spaces_combs_details (pd.DataFrame, optional): spaces' combinations data. Defaults to None.

        Returns:
            pd.DataFrame: decision matrix
        """

        spaces_details = self.__add_num_spaces_var(spaces_details)
        spaces_details = self.__add_distance_var(spaces_details, self.goal_time)

        decisions_variables = ["cancellable_percent", "num_spaces", "distance"]

        return spaces_details[decisions_variables]
