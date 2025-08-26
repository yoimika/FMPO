import abc

class BaseAgent(abc.ABC):
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def sample_action(self):
        pass