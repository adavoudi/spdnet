import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable, Function

import numpy as np


def symmetric(A):
    return 0.5 * (A + A.t())


def is_nan_or_inf(A):
    C1 = torch.nonzero(A == float('inf'))
    C2 = torch.nonzero(A != A)
    if len(C1.size()) > 0 or len(C2.size()) > 0:
        return True
    return False


def is_pos_def(x):
    x = x.cpu().detach().numpy()
    return np.all(np.linalg.eigvals(x) > 0)


class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of 
        Stiefel manifold.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()


class StiefelMetaOptimizer(object):
    """This is a meta optimizer which uses other optimizers for updating parameters
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

        if ctx.needs_input_grad[1]:
            grad_weight = weight.new(input.size(0), weight.size(0), weight.size(1))
            for k, x in enumerate(input):
                g = grad_output[k]
                if len(g.shape) == 1:
                    continue
                P1 = 2 * x.mm(weight.mm(g))
                P2 = weight.mm(weight.t())
                grad_weight[k] = P1 - P2.mm(P1)  
            
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
        mask = np.triu_indices(input.size(1))
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
            grad_input.fill_(0)
            mask_upper = np.triu_indices(input.size(1))
            mask_diag = np.diag_indices(input.size(1))
            for k, g in enumerate(grad_output):
                grad_input[k][mask_upper] = g
                grad_input[k] = grad_input[k] + grad_input[k].t()   
                grad_input[k][mask_diag] /= 2

        return grad_input


class SPDVectorize(nn.Module):

    def __init__(self):
        super(SPDVectorize, self).__init__()

    def forward(self, input):
        return SPDVectorizeFunction.apply(input)

class SPDUnVectorizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        n = int(-.5 + 0.5 * np.sqrt(1 + 8 * input.size(1)))
        output = input.new(len(input), n, n)
        output.fill_(0)
        mask_upper = np.triu_indices(n)
        mask_diag = np.diag_indices(n)
        for k, x in enumerate(input):
            output[k][mask_upper] = x
            output[k] = output[k] + output[k].t()   
            output[k][mask_diag] /= 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            n = int(-.5 + 0.5 * np.sqrt(1 + 8 * input.size(1)))
            grad_input = input.new(len(input), input.size(1))
            mask = np.triu_indices(n)
            for k, g in enumerate(grad_output):
                grad_input[k] = g[mask]

        return grad_input


class SPDUnVectorize(nn.Module):

    def __init__(self):
        super(SPDUnVectorize, self).__init__()

    def forward(self, input):
        return SPDUnVectorizeFunction.apply(input)


class SPDTangentSpaceFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                g = symmetric(g)
                
                s_log_diag = s.log().diag()
                s_inv_diag = (1/s).diag()
                
                dLdV = 2*(g.mm(u.mm(s_log_diag)))
                dLdS = eye * (s_inv_diag.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0
                
                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV)))+dLdS).mm(u.t())


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


class SPDUnTangentSpaceFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.exp_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                g = symmetric(g)
                
                s_exp_diag = s.exp().diag()
                
                dLdV = 2*(g.mm(u.mm(s_exp_diag)))
                dLdS = eye * (s_exp_diag.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0
                
                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV)))+dLdS).mm(u.t())


        return grad_input


class SPDUnTangentSpace(nn.Module):

    def __init__(self, unvectorize=True):
        super(SPDUnTangentSpace, self).__init__()
        self.unvectorize = unvectorize
        if unvectorize:
            self.unvec = SPDUnVectorize()

    def forward(self, input):
        if self.unvectorize:
            input = self.unvec(input)
        output = SPDUnTangentSpaceFunction.apply(input)
        return output



class SPDRectifiedFunction(Function):

    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon].fill_(epsilon[0])
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue

                g = symmetric(g)

                x = input[k]
                u, s, v = x.svd()
                
                max_mask = s > epsilon
                s_max_diag = s.clone(); s_max_diag[~max_mask] = epsilon; s_max_diag = s_max_diag.diag()
                Q = max_mask.diag().float()
                
                dLdV = 2*(g.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV))+dLdS).mm(u.t())
            
        return grad_input, None


class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

    def forward(self, input):
        epsilon = Variable(self.epsilon, requires_grad=False)
        output = SPDRectifiedFunction.apply(input, epsilon)
        return output


class SPDPowerFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s = torch.exp(weight * torch.log(s))
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = None
        grad_weight = None

        eye = input.new(input.size(1))
        eye.fill_(1); eye = eye.diag()
        grad_input = input.new(input.size(0), input.size(1), input.size(2))
        grad_weight = weight.new(input.size(0), weight.size(0))
        for k, g in enumerate(grad_output):
            if len(g.shape) == 1:
                continue

            x = input[k]
            u, s, v = x.svd() 

            g = symmetric(g)
            
            s_log = torch.log(s)
            s_power = torch.exp(weight * s_log)

            s_power = s_power.diag()
            s_power_w_1 = weight * torch.exp((weight-1) * s_log)
            s_power_w_1 = s_power_w_1.diag()
            s_log = s_log.diag()
            
            grad_w = s_log.mm(u.t().mm(s_power.mm(u))).mm(g)
            grad_weight[k] = grad_w.diag()

            dLdV = 2*(g.mm(u.mm(s_power)))
            dLdS = eye * (s_power_w_1.mm(u.t().mm(g.mm(u))))
            
            P = s.unsqueeze(1)
            P = P.expand(-1, P.size(0))
            P = P - P.t()
            mask_zero = torch.abs(P) == 0
            P = 1 / P
            P[mask_zero] = 0            
            
            grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV))+dLdS).mm(u.t())
                
        grad_weight = grad_weight.mean(0)
        
        return grad_input, grad_weight


class SPDPower(nn.Module):

    def __init__(self, input_dim):
        super(SPDPower, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        # nn.init.normal(self.weight)

    def forward(self, input):
        output = SPDPowerFunction.apply(input, self.weight)
        return output
