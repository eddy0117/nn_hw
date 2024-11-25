import numpy as np

class Linear:
    def __init__(self, in_features, out_features, act_func):
        self.in_features = in_features
        self.out_features = out_features
        self.act_func = act_func

        self.w, self.b, self.velocity = self.weight_init()
        self.params_delta = {'dW': None, 'db': None}
        self.activations = {'I': None, 'Y': None}

    def weight_init(self) -> dict:
        velocity = {}
        # 使用 xavier initalization 初始化 weight
        # source: https://www.numpyninja.com/post/weight-initialization-techniques

        # w 是高斯分佈的隨機數，所以使用 xavier init 時分子為 2
        w = np.random.randn(self.in_features, self.out_features) * np.sqrt(2 / self.in_features)
        b = np.zeros((1, self.out_features))
   
        velocity['w'] = np.zeros_like(w)
        velocity['b'] = np.zeros_like(b)
        return w, b, velocity
    
    def update_params(self, opt_params: dict):
        self.velocity['w'] = opt_params['alpha'] * self.velocity['w'] - opt_params['lr'] * self.params_delta['dW']
        self.velocity['b'] = opt_params['alpha'] * self.velocity['b'] - opt_params['lr'] * self.params_delta['db']
        self.w += self.velocity['w']
        self.b += self.velocity['b']
    
    def forward(self, x) -> dict:
        
        I = np.matmul(x, self.w) + self.b
        Y = self.act_func.forward(I)
        self.activations['I'] = I
        self.activations['Y'] = Y
        return Y
    
    def backward(self, back_prop_params: dict = None, is_output=False, label=None):

        if is_output:
            # CrossEntropyLoss
            delta = (self.activations['Y'] - label) * self.act_func.backward(self.activations['I'])
        else:
            delta = np.matmul(back_prop_params['delta_next'], back_prop_params['w_next'].T) * self.act_func.backward(self.activations['I'])
        
        return delta