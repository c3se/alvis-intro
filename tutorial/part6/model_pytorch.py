from torch import nn


class Model(nn.Module):

    def __init__(self, input_size, output_size, verbose=False):
        super(Model, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, 5),
            nn.Linear(5, 5),
            nn.Linear(5, output_size),
        ])
        self.verbose = verbose

    def forward(self, input):
        if self.verbose:
            print("Hello from", input.device, "with input size", input.size())

        x = input
        for layer in self.layers:
            x = layer(x)
        return x
