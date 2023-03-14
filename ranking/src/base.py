from sklearn.base import BaseEstimator, TransformerMixin

from .utils import SpaceDetailsConverter, SpacesCombinationsConverter


class SpaceDetailsConverterPipe(BaseEstimator, TransformerMixin):
    def __init__(self, spaces_data) -> None:
        self.converter = SpaceDetailsConverter(spaces_data)

    def fit(self, X, y=None):
        return self

    def transform(self, X=None):
        return self.converter.get_data()


class SpacesCombinationsConverterPipe(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.converter = SpacesCombinationsConverter()

    def fit(self, X, y=None):
        return self

    def transform(self, X=None):
        return self.converter.get_data()
