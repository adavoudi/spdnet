import numpy as np
import torch

from SPDNet import *

class CTX:
    def __init__(self, saved_variables, needs_input_grad):
        self.saved_variables = saved_variables
        self.needs_input_grad = needs_input_grad

def assertTensorEqual(a, b, tolerance=1e-4):
    return (a.sub(b).abs().max() < tolerance).data.item() == 1

spd = torch.from_numpy(np.asarray([
    [4.2051,1.1989,0.6229],
    [1.1989,4.1973,0.6028],
    [0.6229,0.6028,3.5204]
], np.float32))
spd = spd.unsqueeze(0)

grad_mat = torch.from_numpy(np.asarray([
    [1,1,1],
    [1,1,1],
    [1,1,1]
], np.float32))
grad_mat = grad_mat.unsqueeze(0)

def check_TangentSpace():
    desired_forward = torch.from_numpy(np.asarray([
        [1.3848,0.2849,0.1444],
        [0.2849,1.3837,0.1383],
        [0.1444,0.1383,1.2355]
    ], np.float32))

    desired_backward = torch.from_numpy(np.asarray([
        [ 0.1618,0.1625,0.1885],
        [ 0.1625,0.1631,0.1894],
        [ 0.1885,0.1894,0.2222]
    ], np.float32))

    forward = SPDTangentSpaceFunction.apply(spd)
    backward = SPDTangentSpaceFunction.backward(CTX([spd], [True]), grad_mat)
    
    forward_eq = assertTensorEqual(forward, desired_forward)
    backward_eq = assertTensorEqual(backward, desired_backward)

    return (forward_eq and backward_eq)

def check_Rectified():
    desired_forward = torch.from_numpy(np.asarray([
        [4.2562,1.1498,0.6185],
        [1.1498,4.2443,0.6070],
        [0.6185,0.6070,3.5207]
    ], np.float32))

    desired_backward = torch.from_numpy(np.asarray([
        [0.9991,1.0000,1.0055],
        [1.0000,1.0008,0.9948],
        [1.0055,0.9948,0.9991]
    ], np.float32))

    epsilon = torch.FloatTensor([3.1])

    forward = SPDRectifiedFunction.apply(spd, epsilon)
    backward = SPDRectifiedFunction.backward(CTX([spd, epsilon], [True, False]), grad_mat)[0]
    forward_eq = assertTensorEqual(forward, desired_forward)
    backward_eq = assertTensorEqual(backward, desired_backward)

    return (forward_eq and backward_eq)


def check_Transform():
    desired_forward = torch.from_numpy(np.asarray([
        [5.2656,-0.8131,0.5717],
        [-0.8131,3.4052,-0.3141],
        [0.5717,-0.3141,3.2519]
    ], np.float32))

    desired_backward_net = torch.from_numpy(np.asarray([
        [0.3814,-0.8837,-0.4667],
        [-0.8837,2.0474,1.0814],
        [-0.4667,1.0814,0.5712]
    ], np.float32))

    desired_backward_weight = torch.from_numpy(np.asarray([
        [0.8207,0.8207,0.8207],
        [-11.4424,-11.4424,-11.4424],
        [-6.2772,-6.2772,-6.2772]
    ], np.float32))

    desired_weight = torch.from_numpy(np.asarray([
        [0.6787,0.7195,0.1471],
        [-0.6016,0.6596,-0.4505],
        [-0.4211,0.2173,0.8806]
    ], np.float32))

    weight = torch.from_numpy(np.asarray([
        [-0.4572,0.8640,0.2107],
        [-0.5122,-0.0621,-0.8566],
        [-0.7271,-0.4996,0.4709]
    ], np.float32))

    forward = SPDTransformFunction.apply(spd, weight)
    backward = SPDTransformFunction.backward(CTX([spd, weight], [True, True]), grad_mat)
    forward_eq = assertTensorEqual(forward, desired_forward, tolerance=1e-3)
    backward_eq_1 = assertTensorEqual(backward[0], desired_backward_net, tolerance=1e-3)
    backward_eq_2 = assertTensorEqual(backward[1], desired_backward_weight, tolerance=1e-4)

    grad = orthogonal_projection(backward[1], weight)
    new_weight = retraction(weight, -grad)

    weight_eq = assertTensorEqual(new_weight, desired_weight, tolerance=1e-4)

    return (forward_eq and backward_eq_1 and backward_eq_2 and weight_eq)


units = {
    'Tangent space layer': check_TangentSpace,
    'Rectification layer': check_Rectified,
    'Transformation layer': check_Transform
}

result = True

print('Performing unit test ...')
for index, (name, func) in enumerate(units.items()):
    current_result = func()
    result = result and current_result
    print('[%d/%d] %s : %s' % (index+1, len(units), name, current_result))


if result:
    print('All tests past')
else:
    print('test failed')