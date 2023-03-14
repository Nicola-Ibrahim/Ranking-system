import time

from sklearn.pipeline import Pipeline

from . import base
from .settings import RAW_SPACES_DATA_PATH


def run():
    goal_time = 8  # type: ignore

    # start = time.time()
    # decision_matrix = rank.create_decision_matrix(goal_time=goal_time, data=space_combinations)
    # weights = [69.2, 23.1, 7.7]  # CR = 0%
    # t = rank.Topsis(decision_matrix, weights, criteria=[False, False, False])

    # t.get_rank()
    # end = time.time()
    # print("Execution time:", end - start)

    pipe_line = Pipeline(
        [
            ("space_details_converter", base.SpaceDetailsConverterPipe(RAW_SPACES_DATA_PATH)),
            ("space_combs_details_converter", base.SpacesCombinationsConverterPipe()),
        ]
    )

    data = pipe_line.fit_transform(X=None)
    print(data)
