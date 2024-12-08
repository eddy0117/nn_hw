import numpy as np
from abc import ABC, abstractmethod

class MyModule(ABC):
    def __init__(self):
        self.in_feat = None
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, delta):
        pass
    
    def update_params(self, opt_params):
        pass


# class MyLayerModule(MyModule):
