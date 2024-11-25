import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.activations = {'I': None, 'Y': None}
        self.params_delta = None

    def forward(self, x):
        '''
        x: (N, C, H, W),
        out: (N, C*H*W)
        '''
        self.input_shape = x.shape
        self.output_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))
        self.activations['Y'] = x.reshape(self.output_shape)
        return self.activations['Y']

    def backward(self, back_prop_params):
        '''
        delta_next = delta: (N, n_classes)
        w_next: keep (C*H*W, n_classes)
        '''

        self.w = back_prop_params['w_next']
        return back_prop_params['delta_next']
    
    def update_params(self, opt_params):
        pass