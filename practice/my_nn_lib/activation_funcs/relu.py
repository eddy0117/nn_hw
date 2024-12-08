import numpy as np
from ..core import MyModule

class ReLU(MyModule):
    def forward(self, x):
        self.in_feat = x
        
        return np.maximum(x, 0)

    def backward(self, delta):
        return delta * (self.in_feat > 0)
    
    def update_params(self, opt_params):
        pass
       