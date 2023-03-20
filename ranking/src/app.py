import time

from sklearn.pipeline import Pipeline

from . import settings, transformers, utils


def run():
    goal_time = 8  # type: ignore

    start = time.time()
    # weights = [69.2, 23.1, 7.7]  # CR = 0%
    # t = rank.Topsis(decision_matrix, weights, criteria=[False, False, False])

    # Read spaces data file
    spaces_data = utils.read_json_file(settings.RAW_SPACES_DATA_PATH)

    pipe_line = Pipeline(steps=transformers.pipe_line_steps)
    pipe_line.set_params(make_decision_matrix__kw_args={"goal_time": goal_time})

    data = pipe_line.fit_transform(X=spaces_data)
    end = time.time()
    print("Execution time:", end - start)
