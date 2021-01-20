from abc import ABC, abstractmethod

class IDao(ABC):
    @abstractmethod
    def number_of_training_samples(self):
        pass
