import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable, Function

import numpy as np

eps = 0.00001

def symmetric(A):
    return 0.5 * (A + A.t())

def is_pos_def(x):
    x = x.cpu().detach().numpy()
    return np.all(np.linalg.eigvals(x) > 0)

class StiefelParameter(nn.Parameter):
    r"""A kind of Variable that is to be considered a module parameter on the space of 
        Stiefel manifold.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()


class StiefelMetaOptimizer(object):
    r"""This is a meta optimizer which uses other optimizers for updating parameters
        and remap all StiefelParameter parameters to Stiefel space after they have been updated.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if isinstance(p, StiefelParameter):
                    Q, R = p.data.qr()
                    p.data = Q.clone()
                    

        return loss

class GaussianKernel(nn.Module):

    def __init__(self, kernel_width):
        super(GaussianKernel, self).__init__()
        self.kernel_width = kernel_width

    def forward(self, input):
        output = input.new(input.size(0), input.size(1), input.size(1))
        input = input.view(input.size(0), input.size(1), -1)

        # compute kernel elements for each sample
        for k, x in enumerate(input):
            P1 = x.mm(x.t())
            P2 = P1.diag(0)
            P2 = P2.unsqueeze(1)
            P2 = P2.repeat(1, P2.size(0))
            P2 = P2 + P2.t()
            output[k] = torch.exp(-self.kernel_width*(P2 - 2*P1))

        return output
    

class SPDTransformFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)

        output = input.new(input.size(0), weight.size(1), weight.size(1))
        for k, x in enumerate(input):
            output[k] = weight.t().mm(x.mm(weight))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None

        grad_output[grad_output != grad_output] = 0
        if ctx.needs_input_grad[0]:
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue
                grad_input[k] = weight.mm(g.mm(weight.t()))

            grad_input[grad_input == float('inf')] = 0
            grad_input[grad_input != grad_input] = 0

        if ctx.needs_input_grad[1]:
            grad_weight = weight.new(input.size(0), weight.size(0), weight.size(1))
            for k, x in enumerate(input):
                g = grad_output[k]
                if len(g.shape) == 1:
                    continue
                P1 = 2 * x.mm(weight.mm(g))
                P2 = P1 - weight.mm(weight.t()).mm(P1)
                P2[P2 == float('inf')] = 0
                P2[P2 != P2] = 0
                grad_weight[k] = P2
            
            grad_weight = grad_weight.mean(0)

        return grad_input, grad_weight


class SPDTransform(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDTransform, self).__init__()
        self.output_size = output_size
        self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal(self.weight)

    def forward(self, input):
        return SPDTransformFunction.apply(input, self.weight)


class SPDVectorizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = input.new(input.size(0), input.size(1)*(input.size(1)+1)//2)
        mask = torch.triu(torch.ones(input.size(1), input.size(2))) == 1
        for k, x in enumerate(input):
            output[k] = x[mask]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = input.new(len(input), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                grad_input[k][torch.triu(torch.ones(input.size(1), input.size(2))) == 1] = g
                grad_input[k] = grad_input[k] + grad_input[k].t()
                grad_input[k][torch.eye(input.size(1), input.size(2)) == 1] /= 2

            grad_input[grad_input == float('inf')] = 0
            grad_input[grad_input != grad_input] = 0

        return grad_input


class SPDVectorize(nn.Module):

    def __init__(self):
        super(SPDVectorize, self).__init__()

    def forward(self, input):
        return SPDVectorizeFunction.apply(input)


class SPDTangentSpaceFunction(Function):


    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        rand_spd = torch.randn(input.size(1), input.size(1)) * eps
        rand_spd = rand_spd + rand_spd.t()
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            x = x + rand_spd
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        rand_spd = Variable(torch.randn(input.size(1), input.size(1)) * eps)
        rand_spd = rand_spd + rand_spd.t()

        if ctx.needs_input_grad[0]:
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                x = x + rand_spd
                try:
                    u, s, v = x.svd()
                except:
                    print('Tangent backward.')
                    print('X: ')
                    print(x)
                    exit(0)
                s_log_diag = s.log().diag()
                s_inv_diag = (1/s).diag()
                P = s.unsqueeze(1)
                P = P.repeat(1, P.size(0))
                P = P - P.t()
                mask_zero = P == 0
                P = 1 / P
                P[mask_zero] = 0
                dLdV = 2*(g.mm(u.mm(s_log_diag)))
                dLdS = s_inv_diag.mm(u.t().mm(g.mm(u)))
                P2 = u.mm(symmetric(P.t() * (u.t().mm(dLdV)))+dLdS).mm(u.t())
                P2[P2 == float('inf')] = 0
                P2[P2 != P2] = 0
                grad_input[k] = P2


        return grad_input


class SPDTangentSpace(nn.Module):

    def __init__(self, vectorize=True):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize()

    def forward(self, input):
        output = SPDTangentSpaceFunction.apply(input)
        if self.vectorize:
            output = self.vec(output)

        return output


class SPDRectifiedFunction(Function):

    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        rand_spd = torch.randn(input.size(1), input.size(1)) * eps
        rand_spd = rand_spd + rand_spd.t()
                    
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            x = x + rand_spd
            u, s, v = x.svd()
            s[s < epsilon].fill_(float(epsilon))
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None
        
        rand_spd = Variable(torch.randn(input.size(1), input.size(1)) * eps)
        rand_spd = rand_spd + rand_spd.t()

        if ctx.needs_input_grad[0]:
            Q = input.new(input.size(1), input.size(1))
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue
                x = input[k]
                x = x + rand_spd
                u, s, v = x.svd()
                max_mask = s > epsilon
                s_max_diag = s; s_max_diag[~max_mask] = epsilon; s_max_diag = s_max_diag.diag()
                Q.fill_(0); Q[max_mask] = 1
                P = s.unsqueeze(1)
                P = P.repeat(1, P.size(0))
                P = P - P.t()
                mask_zero = P == 0
                P = 1 / P
                P[mask_zero] = 0
                dLdV = 2*(g.mm(u.mm(s_max_diag)))
                dLdS = Q.mm(u.t().mm(g.mm(u)))
                P2 = u.mm(symmetric(P.t() * (u.t().mm(dLdV)))+dLdS).mm(u.t())
                P2[P2 == float('inf')] = 0
                P2[P2 != P2] = 0
                grad_input[k] = P2
            
        return grad_input, None


class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

    def forward(self, input):
        epsilon = Variable(self.epsilon, requires_grad=False)
        output = SPDRectifiedFunction.apply(input, epsilon)
        return output
