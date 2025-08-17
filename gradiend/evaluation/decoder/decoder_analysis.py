from abc import ABC, abstractmethod


class DecoderAnalysis(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def default_evaluation():
        pass
