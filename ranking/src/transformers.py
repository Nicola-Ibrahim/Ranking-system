from sklearn.base import BaseEstimator, TransformerMixin

from . import rank, utils


class SpaceDetailsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_enc_bits) -> None:
        self.num_enc_bits = num_enc_bits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return utils.SpacesCreator(self.num_enc_bits).create(X)


class SpacesCombinationsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return utils.SpacesCombinationsCreator().create(X)


class DecisionMatrixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, goal_time: int) -> None:
        self.goal_time = goal_time

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return utils.DecisionMatrix(self.goal_time).create(X)


class TopsisTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, weight_matrix: list[int], criteria: list[bool]) -> None:
        self.weight_matrix = weight_matrix
        self.criteria = criteria

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return rank.Topsis(X, self.weight_matrix, self.criteria).get_rank()


pipe_line_steps = [
    ("space_details_parser", SpaceDetailsTransformer(num_enc_bits=24)),
    ("space_combs_details_parser", SpacesCombinationsTransformer()),
    ("decision_matrix", DecisionMatrixTransformer(goal_time=132)),
    ("topsis_rank", TopsisTransformer(weight_matrix=[69.2, 23.1, 7.7], criteria=[False, False, False])),
]
