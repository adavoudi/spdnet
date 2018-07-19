import torch
from torch import nn
from torch.autograd import Variable, Function

import numpy as np

"""
The append mean approach:
    Yu, K., & Salzmann, M. (2017). Second-order convolutional neural networks. arXiv preprint arXiv:1703.06817.
"""
class Covariance(nn.Module):

    def __init__(self, append_mean=True):
        super(Covariance, self).__init__()
        self.append_mean = append_mean

    def forward(self, input):

        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1,2)) / input.size(1)
        
        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1,2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1,1,1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1,2)
            output = torch.cat((output, mean), 1)

        return output


class PolynomialKernel(nn.Module):

    def __init__(self, degree=1, added_value=1):
        super(PolynomialKernel, self).__init__()
        
        self.degree = degree
        self.added_value = added_value

    def forward(self, input):
        """
        Arguments:
            input {Tensor} -- A 3-D tensor of size :math:`(N, C, M)`
        
        Returns:
            [Tensor] -- :math:`N` SPD matrices of size :math:`(C, C)`, i.e. :math:`(N, C, C)`
        """

        output = torch.bmm(input, input.transpose(1,2))
        output.add_(self.added_value)
        output.pow_(self.degree)

        return output


class GaussianKernel(nn.Module):

    def __init__(self, in_channels, kernel_width=None, laplacian_kernel=False):
        """
        Arguments:
            kernel_width {float} -- -0.5/sigma^2 if 
        
        Keyword Arguments:
            laplacian_kernel {bool} -- Whether to use Laplacian kernel (default: {False})
        """

        super(GaussianKernel, self).__init__()

        self.kernel_width = kernel_width
        self.laplacian_kernel = laplacian_kernel
        self.in_channels = in_channels

        self.register_buffer('diag_idx', torch.LongTensor([t for t in range(self.in_channels)]))
    
        self.var_kernel = kernel_width is None
        if self.var_kernel:
            self.kernel_width = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
            nn.init.constant(self.kernel_width, 1)

    def forward(self, input):
        """
        Arguments:
            input {Tensor} -- A 3-D tensor of size :math:`(N, C, M)`
        
        Returns:
            [Tensor] -- :math:`N` SPD matrices of size :math:`(C, C)`, i.e. :math:`(N, C, C)`
        """

        kernel_width = self.kernel_width.abs() if self.var_kernel else self.kernel_width
        P1 = torch.bmm(input, input.transpose(1,2))
        P2 = P1[:, self.diag_idx, self.diag_idx]
        P2 = P2.unsqueeze(2).expand(-1, -1, input.size(1))
        P2.add_(P2.transpose(1,2))
        P2.add_(-2 * P1)
        if self.laplacian_kernel:
            P2.sqrt_()
        P2.mul_(-1* kernel_width)
        P2.exp_()
        
        return P2
 

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

        weight_a = self.weight_a if self.use_weight_for_a else 1
        weight_b = self.weight_b if self.use_weight_for_b else 1

        if self.is_product:
            output = weight_a * weight_b * input1 * input2
        else:
            output = weight_a * input1 + weight_b * input2

        return output
