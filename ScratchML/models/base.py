from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
