import torch
from torch import nn
import numpy as np
import math
import random

from .nn import NN
from .. import activations
from .. import initializers
from ... import config


# Define input encoding function
L= 0.06 #1.0
M=1.


def xavier_init(d_in, d_out):
    glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
    W = glorot_stddev * torch.randn((d_in, d_out),  dtype=torch.float32)
    b = torch.zeros(d_out ,dtype=torch.float32)
    return W, b

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def init_layer(d_in, d_out):
    W, b = xavier_init(d_in, d_out)
    return W, b


class MMLP(NN):
    def __init__(self, layers, activation, kernel_initializer = None):
        super(MMLP, self).__init__()

        self.activation = activations.get(activation)

        L_x = 0.06
        L_y = 0.04
        M_x, M_y = 1, 1
        w_x = 2.0 * math.pi / L_x
        w_y = 2.0 * math.pi / L_y
        k_x = torch.arange(1, M_x + 1, dtype = torch.float32)
        k_y = torch.arange(1, M_y + 1, dtype = torch.float32)
        k_xx, k_yy = torch.meshgrid(k_x, k_y)
        k_xx = k_xx.flatten()
        k_yy = k_yy.flatten()
        self.d0 = layers[0] #2 * M_x + 2 * M_y + 4 * M_x * M_y +1
        self.enc_params = {"w_x": w_x,
                           "w_y": w_y,
                           "k_x": k_x.reshape(1, -1),
                           "k_y": k_y.reshape(1, -1),
                           "k_xx": k_xx.reshape(1, -1),
                           "k_yy": k_yy.reshape(1, -1)}


        self.U1, self.b1 = xavier_init(self.d0 , layers[1])
        self.U2, self.b2 = xavier_init(self.d0 , layers[1])
        # self.params = nn.ModuleList([init_layer(d_in, d_out) for d_in, d_out in zip(layers[:-1], layers[1:])])

        def make_linear(n_input, n_output):
            linear = torch.nn.Linear(n_input, n_output, dtype=config.real(torch))
            # initializer(linear.weight)
            # initializer_zero(linear.bias)
            return linear

        self.params = torch.nn.ModuleList()
        for layer_idx, (d_in, d_out) in enumerate(zip(layers[:-1], layers[1:])):
            if layer_idx == 0:
                d_in = self.d0
            self.params.append(make_linear(d_in, d_out))
        self.params.apply(init_weights)


        self.freq = torch.randn( (layers[0], layers[1] // 2), dtype=torch.float32)
    def input_encoding(self, x):
        a = (x[:,0:1]/0.06)**2 + (x[:,1:2]/0.04)**2 - 1
        out = torch.cat([ torch.ones_like(x[:,0:1]),
                          a,
                          x[:,0:1],
                          x[:,1:2]
                          # torch.cos(x[:,0:1]/0.06),
                          # torch.sin(x[:,1:2]/0.04),
                          # torch.sin(x[:, 0:1] / 0.06),
                          # torch.cos(x[:, 1:2] / 0.04),
                          # torch.cos(x[:, 1:2] / 0.04) * torch.sin(x[:, 0:1] / 0.06),
                          # torch.cos(x[:, 1:2] / 0.04) * torch.cos(x[:, 0:1] / 0.06),
                          # torch.sin(x[:, 1:2] / 0.04) * torch.sin(x[:, 0:1] / 0.06),
                          # torch.sin(x[:, 1:2] / 0.04) * torch.cos(x[:, 0:1] / 0.06),
                         ], dim = -1)
        # print(torch.cos(x[:, 0:1] @ self.enc_params['k_xx'] * self.enc_params['w_x']).shape, torch.cos(
        #                      x[:, 1:2] @ self.enc_params['k_yy'] * self.enc_params['w_y']).shape)
        # out = torch.cat([torch.ones((x.shape[0], 1)), torch.asin(x[:, 0:1] @ self.enc_params['k_x'] *  self.enc_params['w_x']),
        #                  torch.acos(x[:, 0:1] @ self.enc_params['k_x'] * self.enc_params['w_x']),
        #                  torch.asin( x[:, 1:2] @ self.enc_params['k_x'] *self.enc_params['w_x']),
        #                  torch.acos(x[:, 1:2] @ self.enc_params['k_x'] * self.enc_params['w_x']),
        #                  torch.acos(x[:, 0:1] @ self.enc_params['k_xx'] * self.enc_params['w_x']) * torch.acos(
        #                      x[:, 1:2] @ self.enc_params['k_yy'] * self.enc_params['w_y']),
        #                  torch.acos(x[:, 0:1] @ self.enc_params['k_xx'] * self.enc_params['w_x']) * torch.asin(
        #                      x[:, 1:2] @ self.enc_params['k_yy'] * self.enc_params['w_y']),
        #                  torch.asin(x[:, 0:1] @ self.enc_params['k_xx'] *  self.enc_params['w_x']) * torch.acos(
        #                      x[:, 1:2] @ self.enc_params['k_yy'] * self.enc_params['w_y']),
        #                  torch.asin(x[:, 0:1] @ self.enc_params['k_xx'] *  self.enc_params['w_x']) * torch.asin(
        #                      x[:, 1:2] @ self.enc_params['k_yy'] * self.enc_params['w_y']),
        #                  ], dim = -1)

        return out

    def forward(self, x):
        # inputs = self.input_encoding(x)
        inputs = x * 1.0
        U = self.activation(torch.mm(inputs, self.U1) + self.b1)
        V = self.activation(torch.mm(inputs, self.U2) + self.b2)

        for ly in self.params[:-1]:
            outputs = self.activation(ly(inputs))
            inputs = torch.mul(outputs, U) + torch.mul(1 - outputs, V)
        outputs = self.params[-1](inputs)

        return outputs

