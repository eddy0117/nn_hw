import numpy as np

class ReLU:
    def forward(x):
        return np.maximum(x, 0)
    def backward(x):
        return x > 0
    
class LeckyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    def backward(self, x):
        return np.where(x > 0, 1, self.alpha)

class Softmax:
    def forward(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    def backward(x):
        # 限搭配 CrossEntropyLoss 使用
        # 將 Softmax 與 CrossEntropyLoss 的導數合併在 delta 項計算
        # 所以 activateion 項係數為 1
        return 1

class Identity:
    def forward(x):
        return x
    def backward(x):
        return 1