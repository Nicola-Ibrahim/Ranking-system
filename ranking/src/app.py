import time

from sklearn.pipeline import Pipeline

from . import settings, transformers, utils


def run():
    start = time.time()

    # Read spaces data file
    spaces_data = utils.read_json_file(settings.RAW_SPACES_DATA_PATH)

    pipe_line = Pipeline(steps=transformers.pipe_line_steps)
    # pipe_line.set_params(add_distance_variable__kw_args={"goal_time": goal_time})

    data = pipe_line.fit_transform(X=spaces_data)
    print(data)

    end = time.time()
    print("Execution time:", end - start)
