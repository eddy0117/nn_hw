import numpy as np


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, act_func=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.act_func = act_func
        self.w, self.b, self.velocity = self.weight_init()
        self.params_delta = {'dW': None, 'db': None}
        self.activations = {'I': None, 'Y': None}
        self.im2col_feat = None
    def weight_init(self) -> dict:
        params = {}
        velocity = {}
        # 使用 xavier initalization 初始化 weight
        # source: https://www.numpyninja.com/post/weight-initialization-techniques

        # w 是高斯分佈的隨機數，所以使用 xavier init 時分子為 2
        kh, kw = self.kernel_size
        # w = np.random.randn(self.out_channels, self.in_channels, kh, kw) * np.sqrt(2 / (self.in_channels * kh * kw))
        w = np.random.randn(self.in_channels * kh * kw, self.out_channels) * np.sqrt(2 / (self.in_channels * kh * kw))
        b = np.zeros((1, self.out_channels))
  
        velocity['w'] = np.zeros_like(w)
        velocity['b'] = np.zeros_like(b)
        return w, b, velocity
    
    def update_params(self, opt_params: dict):
        self.velocity['w'] = opt_params['alpha'] * self.velocity['w'] - opt_params['lr'] * self.params_delta['dW']
        self.velocity['b'] = opt_params['alpha'] * self.velocity['b'] - opt_params['lr'] * self.params_delta['db']
        self.w += self.velocity['w']
        self.b += self.velocity['b']
    
    def forward(self, x) -> dict:
        
        I = self.convolution(x)
        Y = self.act_func.forward(I)
        self.activations['I'] = I
        self.activations['Y'] = Y
        return Y
    
    def backward(self, back_prop_params: dict = None):
        #  如果 next layer 是 flatten
        #  delta_next -> (N, n_classes)
        #  next_w.T: (n_classes, out_c*out_h*out_w)
        #  delta (∂L/∂Z) -> (N, out_c*out_h*out_w)

        #  如果 next layer 是 Conv2d, 
        #  im2col_feat_next -> (N*out_h*out_w, in_c*kh*kw)
        #  delta_next 要從 (N, out_c*out_h*out_w) reshape 成 (N*out_h*out_w, out_c)
        #  next_w -> (out_c, C*kh*kw) 
        #  delta -> (N*out_h*out_w, C*kh*kw)
        

        if len(back_prop_params['w_next'].shape) == 4:
            N, out_c, out_h, out_w = back_prop_params['delta_next'].shape
            back_prop_params['delta_next'] = back_prop_params['delta_next'].reshape(N, out_c, out_h, out_w)\
            .transpose(0, 2, 3, 1)\
            .reshape(N * out_h * out_w, out_c)
     
            

        delta = np.matmul(back_prop_params['delta_next'], back_prop_params['w_next'].T) * self.act_func.backward(back_prop_params['im2col_feat_next'])
        
        return delta
    
    def im2col(self, input_feat: np.ndarray, N, kh, kw, out_h, out_w, stride):
        im2col_feat = []
        for n in range(N):
            for ih in range(out_h):
                for iw in range(out_w):
                    im2col_feat.append(input_feat[n, :, stride * ih:stride * ih + kh, stride * iw:stride * iw + kw])
                    # each element -> (C, kh, kw)
        # input_feat -> (N*out_h*out_w, C, kh, kw)

        return np.array(im2col_feat).reshape(N * out_h * out_w, -1)

    def convolution(self, input_feat: np.ndarray):
        '''
        input_feat: (N, C, H, W)
        filter: (out_c, C*kh*kw)
        bias: (out_C, 1)
        '''
        N, C, H, W = input_feat.shape
        kh, kw = self.kernel_size
        out_h = int((H - kh + 2 * self.padding) // self.stride) + 1
        out_w = int((W - kw + 2 * self.padding) // self.stride) + 1
        out_c = self.w.shape[0]
        
        if self.padding:
            input_feat = np.pad(input_feat, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)

        self.im2col_feat = self.im2col(input_feat, N, kh, kw, out_h, out_w, self.stride)
        # im2col -> (N*out_h*out_w, C*kh*kw)

        # self.w = self.w.reshape(out_c, -1)
        # filter -> (out_c, C*kh*kw)

        # x @ w
        # x-> (N*out_h*out_w, C*kh*kw)
        # w -> (C*kh*kw, out_c)
        if isinstance(self.b, np.ndarray):
            out_feat = (self.im2col_feat @ self.w).T + self.b
        else:
            out_feat = (self.im2col_feat @ self.w).T
        # out_feat -> (out_c, N*out_h*out_w)
        
        # 將 w 重新 reshape 成 (out_c, in_c, kh, kw)
        # self.w = self.w.reshape(out_c, C, kh, kw)

        # 直接將 (out_c, N*out_h*out_w) reshape 成 (N, out_c, out_h, out_w) 會產生順序錯亂
        # 所以先將 (out_c, N*out_h*out_w) 拆成 (out_c, N, out_h, out_w) 後再 permute
        # out_feat -> (N, out_c, out_h, out_w)
        return out_feat.reshape(out_c, N, out_h, out_w).transpose(1, 0, 2, 3)
        # return out_feat.T.reshape(N, out_h, out_w, out_c).transpose(0, 3, 1, 2)