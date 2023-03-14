from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from .rank import create_decision_matrix
from .utils import SpaceDetailsConverter, SpacesCombinationsConverter


class SpaceDetailsConverterPipe(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return SpaceDetailsConverter(X).get_data()


class SpacesCombinationsConverterPipe(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X=None):
        return SpacesCombinationsConverter().get_data()


pipe_line_steps = [
    ("space_details_converter", SpaceDetailsConverterPipe()),
    ("space_combs_details_converter", SpacesCombinationsConverterPipe()),
    ("make_decision_matrix", FunctionTransformer(create_decision_matrix, kw_args={"goal_time": 8})),
]
