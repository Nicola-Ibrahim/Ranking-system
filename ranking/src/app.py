import time

from sklearn.pipeline import Pipeline

from . import settings, transformers


def run():
    goal_time = 8  # type: ignore

    start = time.time()
    # weights = [69.2, 23.1, 7.7]  # CR = 0%
    # t = rank.Topsis(decision_matrix, weights, criteria=[False, False, False])

    pipe_line = Pipeline(steps=transformers.pipe_line_steps)
    data = pipe_line.fit_transform(X=settings.RAW_SPACES_DATA_PATH)
    print(data)
    end = time.time()
    print("Execution time:", end - start)
