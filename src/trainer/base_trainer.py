import abc 

class BaseTrainer(abc.ABC):
    @abc.abstractmethod
    def train(self):
        pass

    # @abc.abstractmethod
    # def evaluate(self):
    #     pass
