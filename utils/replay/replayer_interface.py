from abc import *

class ReplayInterface(metaclass=ABCMeta):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def push(self, transition):
        pass

    @abstractmethod
    def batch_replay(self, batch_size: int):
        pass
