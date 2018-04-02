import torch
from torch import nn
from torch.autograd import Variable, Function

import numpy as np

def is_nan_or_inf(A):
    C1 = torch.nonzero(A == float('inf'))
    C2 = torch.nonzero(A != A)
    if len(C1.size()) > 0 or len(C2.size()) > 0:
        return True
    return False

class PolynomialKernel(nn.Module):

    def __init__(self, num_input_features, use_center=False, degree=1, added_value=0, center_init_scale=1):
        super(PolynomialKernel, self).__init__()
        
        self.degree = degree
        self.added_value = added_value
        self.num_input_features = num_input_features
        self.use_center = use_center

        if use_center:
            self.center = nn.Parameter(torch.FloatTensor(num_input_features), requires_grad=True)
            nn.init.uniform(self.center, a=-1*center_init_scale, b=1*center_init_scale)


    def forward(self, input):
        """
        Arguments:
            input {Tensor} -- A 3-D tensor of size :math:`(N, C, M)`
        
        Returns:
            [Tensor] -- :math:`N` SPD matrices of size :math:`(C, C)`, i.e. :math:`(N, C, C)`
        """
        output = input.new(input.size(0), self.num_input_features, self.num_input_features)
        if self.use_center:
            center = self.center.unsqueeze(1)
            center = center.expand(-1, input.size(2))

        for k, x in enumerate(input):
            if self.use_center:
                P = x.mm(center.t())
            else:
                P = x.mm(x.t())
            
            P = P.add(self.added_value)
            P = P.pow(self.degree)
            output[k] = P

        return output


class GaussianKernel(nn.Module):

    def __init__(self, num_input_features, use_center=False, kernel_width=None, laplacian_kernel=False, center_init_scale=1):
        """
        Arguments:
            kernel_width {float} -- -0.5/sigma^2 if 
        
        Keyword Arguments:
            laplacian_kernel {bool} -- Whether to use Laplacian kernel (default: {False})
        """

        super(GaussianKernel, self).__init__()

        self.num_input_features = num_input_features
        self.use_center = use_center
        self.kernel_width = kernel_width
        self.laplacian_kernel = laplacian_kernel
        
        if kernel_width is None:
            self.kernel_width = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
            nn.init.constant(self.kernel_width, 1)
        
        if use_center:
            self.center = nn.Parameter(torch.FloatTensor(num_input_features), requires_grad=True)
            nn.init.uniform(self.center, a=-1*center_init_scale, b=1*center_init_scale)

    def forward(self, input):
        """
        Arguments:
            input {Tensor} -- A 3-D tensor of size :math:`(N, C, M)`
        
        Returns:
            [Tensor] -- :math:`N` SPD matrices of size :math:`(C, C)`, i.e. :math:`(N, C, C)`
        """

        output = input.new(input.size(0), self.num_input_features, self.num_input_features)
        if self.use_center:
            center = self.center.unsqueeze(1)
            center = center.expand(-1, input.size(2))

        for k, x in enumerate(input):
            if self.use_center:
                P1 = x.mm(center.t())
            else:
                P1 = x.mm(x.t())
            
            P2 = P1.diag(0)
            P2 = P2.unsqueeze(1)
            P2 = P2.expand(-1, P2.size(0))
            P2 = P2 + P2.t()
            P2 = P2 - 2*P1
            if self.laplacian_kernel:
                P2 = P2.sqrt()
            output[k] = torch.exp(-1 * self.kernel_width*P2)

        return output
 

class MixKernel(nn.Module):

    def __init__(self, is_product=False, use_weight_for_a=False, use_weight_for_b=False):

        super(MixKernel, self).__init__()
        self.is_product = is_product
        self.use_weight_for_a = use_weight_for_a
        self.use_weight_for_b = use_weight_for_b
        if use_weight_for_a:
            self.weight_a = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        if use_weight_for_b:
            self.weight_b = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        

    def forward(self, input1, input2):
        """
        Arguments:
            input1 {Tensor} -- A 3-D tensor of size :math:`(N, C, C)`
            input2 {Tensor} -- A 3-D tensor of size :math:`(N, C, C)`
        
        Returns:
            [Tensor] -- :math:`N` SPD matrices of size :math:`(C, C)`, i.e. :math:`(N, C, C)`
        """

        output = input1.new(input1.size(0), input1.size(1), input1.size(1))

        for k in range(input1.size(0)):
            x1 = input1[k]
            x2 = input2[k]
            if self.use_weight_for_a:
                x1 = x1 * self.weight_a
            if self.use_weight_for_b:
                x2 = x2 * self.weight_b

            if self.is_product:
                output[k] = x1 * x2
            else:
                output[k] = x1 + x2

        return output
