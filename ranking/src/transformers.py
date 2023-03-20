from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from . import rank, utils


class SpaceDetailsParserPipe(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return utils.SpaceDetailsParser().get_data(X)


class SpacesCombinationsParserPipe(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return utils.SpacesCombinationsParser().get_data(X)


pipe_line_steps = [
    ("space_details_parser", SpaceDetailsParserPipe()),
    ("space_combs_details_parser", SpacesCombinationsParserPipe()),
    ("make_decision_matrix", FunctionTransformer(rank.create_decision_matrix)),
]
