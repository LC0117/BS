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
    

torch.manual_seed(1)
input = torch.randn(50,50,dtype=torch.double,requires_grad=True)
print(input.sigmoid())
counts = 0
for i in range(100000):
    linear = BSActicvation.apply(input)
    if linear[0][0] == 1:
        counts += 1

print(counts/100000)

# output = BSActicvation.apply(input)
# print(output)
# output.backward(torch.ones_like(output))
# print(input.grad)