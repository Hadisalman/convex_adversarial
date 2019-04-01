import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io as sio

import numpy as np
import cvxpy as cp

import argparse

import os

"""
Converts saved `.pth` files produced by authors' code to `.mat` files in a
format we can process.
"""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

lpnet_torch = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert .pth checkpoint file from pytorch training for WK17a networks.')
    parser.add_argument('name', help='name of .pth file to be converted')

    args = parser.parse_args()

    name = os.path.splitext(args.name)[0]

    lpnet_torch.load_state_dict(torch.load("{}.pth".format(name), map_location="cpu"))

    parameters_torch=dict()
    # transposing the tensor is necessary because pytorch and Julia have different conventions.
    parameters_torch["conv1/weight"] = np.transpose(lpnet_torch[0].weight.data.numpy(), [2, 3, 1, 0])
    parameters_torch["conv1/bias"] = lpnet_torch[0].bias.data.numpy()
    parameters_torch["conv2/weight"] = np.transpose(lpnet_torch[2].weight.data.numpy(), [2, 3, 1, 0])
    parameters_torch["conv2/bias"] = lpnet_torch[2].bias.data.numpy()
    parameters_torch["fc1/weight"] = np.transpose(lpnet_torch[5].weight.data.numpy())
    parameters_torch["fc1/bias"] = lpnet_torch[5].bias.data.numpy()
    parameters_torch["logits/weight"] = np.transpose(lpnet_torch[7].weight.data.numpy())
    parameters_torch["logits/bias"] = lpnet_torch[7].bias.data.numpy()

    sio.savemat("{}.mat".format(name), parameters_torch)
