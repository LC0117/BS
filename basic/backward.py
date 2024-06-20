import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.autograd.gradcheck
from random import random


class BSActicvation(Function):
    @staticmethod
    def forward(ctx: torch.Any, input):
        ctx.save_for_backward(input)
        result = (input.sigmoid()).bernoulli()
        return result

    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Any) -> torch.Any:
        input, = ctx.saved_tensors
        sigmoid_input = input.sigmoid()
        result = grad_output * (sigmoid_input * (1 - sigmoid_input)).bernoulli()
        print(result)
        return result
    
N = 100000
torch.manual_seed(1)
input = torch.randn(50,50,dtype=torch.double,requires_grad=True)
print(input.sigmoid())
counts = 0
for i in range(N):
    linear = BSActicvation.apply(input)
    if linear[0][0] == 1:
        counts += 1

print(counts/N)

# output
# >>> tensor([[0.4228, 0.3289, 0.3254,  ..., 0.6799, 0.5629, 0.3011],
#         [0.7652, 0.6471, 0.2883,  ..., 0.2076, 0.1577, 0.0996],
#         [0.9267, 0.2632, 0.7276,  ..., 0.3984, 0.5138, 0.2436],
#         ...,
#         [0.6362, 0.9650, 0.5181,  ..., 0.5405, 0.4011, 0.2575],
#         [0.1292, 0.2215, 0.7818,  ..., 0.5221, 0.1867, 0.5998],
#         [0.5571, 0.6172, 0.3321,  ..., 0.7459, 0.5479, 0.4279]],
#        dtype=torch.float64, grad_fn=<SigmoidBackward0>)
# 0.42396