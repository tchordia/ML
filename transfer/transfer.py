import torch
from torch import Tensor
from torch.nn.parameter import Parameter


class ExtendibleLinear(torch.nn.Linear):
    def updateUniverseSize(self, n):
        f_to_add = n - self.in_features
        self.weights = torch.cat(
            (self.weights, Parameter(torch.Tensor(self.out_features, f_to_add))), 1)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """

        # The hidden size is between the input and the output size
        w_in = 1
        w_out = 5
        H = (D_in * w_in + D_out * w_out) / (w_in + w_out)

        super(TwoLayerNet, self).__init__()
        self.linear1 = ExtendibleLinear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.D_out = D_out

    def updateUniverseSize(self, D_in):
        self.linear1.updateUniverseSize(D_in)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


initial_input_size = 5
initial_output_size = 5

# Number of timesteps back to look
T = 10

I = initial_input_size
U = (initial_input_size + initial_output_size) * T
O = initial_output_size

m1 = TwoLayerNet(U, O)
models = [m1]

initialInput = torch.rand(I, 1)
initialOutput = torch.sin(initialInput)

for x in range(100):
    # forward pass
    y_pred = m1()
