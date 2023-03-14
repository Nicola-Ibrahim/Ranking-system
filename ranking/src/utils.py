import collections
import itertools
import json
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd

from . import parsing, settings


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
        file = json.load(f, object_hook=parsing.decode_date_time)

    return file


def save_to_json(file_path):
    def inner(func: Callable[[str], dict]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            spaces_combs_details = func(*args, **kwargs)

            # Save to json file
            with open(file_path, mode="w") as f:
                json.dump(spaces_combs_details, f, indent=4, cls=parsing.DateTimeEncoder)

            return spaces_combs_details

        return wrapper

    return inner


class SpaceDetailsConverter:
    def __init__(self, spaces_data: dict = None) -> None:
        # Read spaces data file
        if spaces_data is None:
            self.spaces_data = spaces_data

        else:
            with open(settings.RAW_SPACES_DATA_PATH, mode="r") as f:
                self.spaces_data = json.load(f, object_hook=parsing.decode_date_time)

        self.spaces_details = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
        )

    def encode_spaces_time_range(self) -> dict:
        """Encode available dates' time for each space

        Args:
            time_span (int): the search range

        Returns:
            dict: encoded time for each space-date pair
        """

        for space, details in self.spaces_data.items():
            for available_date in details["available_dates"]:
                date_as_key = available_date["start"].date().strftime("%d/%m/%Y")

                # Assign 0 bits list
                if not self.spaces_details[space]["dates"][date_as_key]["enc"]:
                    self.spaces_details[space]["dates"][date_as_key]["enc"] = [0] * settings.NUM_OF_ENCODED_BITS

                end_time = available_date["end"].hour
                if available_date["end"].hour >= 23 and available_date["end"].minute > 0:
                    end_time = available_date["end"].hour + 1

                # Flip bits where there is free time
                for i in range(available_date["start"].hour, end_time):
                    self.spaces_details[space]["dates"][date_as_key]["enc"][i] = 1

                # Assign 0 value to time range

                self.spaces_details[space]["dates"][date_as_key]["time_range"] = sum(
                    self.spaces_details[space]["dates"][date_as_key]["enc"]
                )

            self.spaces_details[space]["cancellable"] = details["cancellable"]

    @save_to_json(settings.PRE_PROC_SPACES_TIME_PATH)
    def get_data(self) -> dict:
        """Restructure the spaces details


        Returns:
            dict: spaces data
        """

        # Encode the spaces; time range

        self.encode_spaces_time_range()
        return self.spaces_details


class SpacesCombinationsConverter:
    def __init__(self, spaces_data: dict = None) -> None:
        self.spaces_combs_details = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(dict))
        )
        # Read spaces data file
        self.encoded_spaces_data = (
            read_json_file(settings.PRE_PROC_SPACES_TIME_PATH) if spaces_data is None else spaces_data
        )

        # Encode the spaces; time range

    def __create_space_combinations(self, spaces: list) -> list:
        """Create combinations from the available spaces

        Args:
            spaces (list): spaces list

        Returns:
            list: list of created combinations
        """
        combinations_lst = list(
            itertools.chain.from_iterable(itertools.combinations(spaces, r) for r in range(len(spaces) + 1))
        )
        # Remove the first empty element [(),]
        combinations_lst = combinations_lst[1:]

        return combinations_lst

    def __get_cancellable_percent(self, cancellables: list) -> float:
        """Get the percentage of cancellable spaces from the combination

        Args:
            cancellables (dict): cancellable values

        Returns:
            float: cancellable percentage value
        """
        return sum(cancellables) / len(cancellables)

    def __bitwise_time_ranges(self, time_ranges: np.ndarray) -> list:
        """Do bitwise operator (OR) for binary time ranges

        Args:
            time_ranges (np.ndarray): time ranges for each combination

        Returns:
            list: squashed time ranges
        """
        result = np.zeros(shape=[1, settings.NUM_OF_ENCODED_BITS], dtype=int).flatten()
        for time_range in time_ranges:
            result |= np.array(time_range)

        return result.tolist()

    def __get_unique_dates(self, values):
        # Get unique dates
        dates = set()
        for details in values:
            dates.update(list(details["dates"].keys()))

        dates = sorted(dates)

        return dates

    def __merge_combination_time(self, combination_details: dict) -> dict:
        """Merge combination spaces' time range

        Args:
            combination_details (dict): spaces' detail in each combination

        Returns:
            dict: merged encoded time range
        """

        comb_keys = list(combination_details.keys())

        # If there is only one space in the combination
        if len(comb_keys) < 2:
            # Return the sames corresponding space details to the combination
            return list(combination_details.values())[0]["dates"]

        # If there is many spaces in the combination
        # Then, squash the spaces dates into one combination

        combs_details = collections.defaultdict(lambda: collections.defaultdict(dict))

        dates = self.__get_unique_dates(combination_details.values())
        for date in dates:
            summing_date = [combination_details[key]["dates"][date]["enc"] for key in combination_details.keys()]

            combs_details[date]["enc"] = self.__bitwise_time_ranges(summing_date)

            if not combs_details[date]["time_range"] and isinstance(combs_details[date]["time_range"], dict):
                combs_details[date]["time_range"] = 0

            combs_details[date]["time_range"] += sum(combs_details[date]["enc"])
        return combs_details

    def __squash_spaces_combination_details(self) -> dict:
        """Squash available dates' time for each space

        Args:
            spaces_data (dict): encoded spaces' details

        Returns:
            dict: encoded time for each combination
        """
        # TODO: sorting the the spaces by "hours" before create combinations

        combinations = self.__create_space_combinations(self.encoded_spaces_data.keys())
        max_combination_size = len(combinations[-1])

        for comb in combinations:
            comb_to_str = ",".join(comb)

            # Get the cancellable spaces
            self.spaces_combs_details[comb_to_str]["cancellable_spaces_percentage"] = self.__get_cancellable_percent(
                [self.encoded_spaces_data[space]["cancellable"] for space in comb]
            )

            self.spaces_combs_details[comb_to_str]["num_cancellable_spaces"] = sum(
                [self.encoded_spaces_data[space]["cancellable"] for space in comb]
            )

            # Get the spaces
            self.spaces_combs_details[comb_to_str]["spaces_percentage"] = len(comb) / max_combination_size

            self.spaces_combs_details[comb_to_str]["num_spaces"] = len(comb)

            # Merging spaces' time in each combination
            self.spaces_combs_details[comb_to_str]["dates"] = self.__merge_combination_time(
                {i: self.encoded_spaces_data[i] for i in comb}
            )

            # Get total time range\
            for date in self.spaces_combs_details[comb_to_str]["dates"]:
                if not self.spaces_combs_details[comb_to_str]["total_time_range"]:
                    self.spaces_combs_details[comb_to_str]["total_time_range"] = 0

                self.spaces_combs_details[comb_to_str]["total_time_range"] += self.spaces_combs_details[comb_to_str][
                    "dates"
                ][date]["time_range"]

    @save_to_json(settings.PROC_SPACES_DATA_PATH)
    def get_data(self) -> dict:
        """Get the encoded spaces' combination detail

        Returns:
            dict: combination of spaces data
        """

        self.__squash_spaces_combination_details()
        return pd.DataFrame(self.spaces_combs_details).T
