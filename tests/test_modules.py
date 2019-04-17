import numpy as np
import torch

from spdnet.spd import * 

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

def check_UnTangentSpace():
    tang = SPDTangentSpaceFunction.apply(spd)
    untang = SPDUnTangentSpaceFunction.apply(tang)

    transform_assert = assertTensorEqual(spd, untang, tolerance=1e-4)
    return transform_assert

units = {
    'Tangent space layer': check_TangentSpace,
    'Rectification layer': check_Rectified,
    'Untangent space layer': check_UnTangentSpace
}

result = True

print('Performing unit test ...')
for index, (name, func) in enumerate(units.items()):
    current_result = func()
    result = result and current_result
    print('[%d/%d] %s : %s' % (index+1, len(units), name, current_result))


if result:
    print('All tests passed')
else:
    print('test failed')