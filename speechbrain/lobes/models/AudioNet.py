"""A simple audio classification architecture for AudioMNIST https://arxiv.org/abs/1807.03418.

Authors
 * Nicolas Aspert 2024
"""

# import os
import torch  # noqa: F401
import torch.nn as nn
from torch.nn import Dropout, Flatten

import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.pooling import Pooling1d


class AudioNet(nn.Module):
    """This model extracts features for spoken digits recognition.

    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time-delay neural (TDNN) layers.
    tdnn_channels : list of ints
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dilations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.
    in_channels : int
        Expected size of input features.

    Example
    -------
    >>> compute_xvect = Xvector('cpu')
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = compute_xvect(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        activation=torch.nn.ReLU,
        conv_blocks=3,
        conv_channels=[64, 128, 128],
        conv_kernel_sizes=[3, 3, 3],
        conv_dilations=[
            1,
            1,
            1,
        ],
        in_channels=40,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.flatten = Flatten()
        # Conv layers
        for block_index in range(conv_blocks):
            out_channels = conv_channels[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=conv_kernel_sizes[block_index],
                        dilation=conv_dilations[block_index],
                    ),
                    activation(),
                    Pooling1d(kernel_size=2, pool_type="max", stride=2),
                ]
            )
            in_channels = conv_channels[block_index]

    def forward(self, x, lens=None):
        """Returns the x-vectors.

        Arguments
        ---------
        x : torch.Tensor
            Inputs features for extracting x-vectors.
        lens : torch.Tensor
            The corresponding relative lengths of the inputs.

        Returns
        -------
        x : torch.Tensor
            X-vectors.
        """

        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)
        return self.flatten(x)


class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of xvector features.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_xvect = Xvector()
    >>> xvects = compute_xvect(input_feats)
    >>> classify = Classifier(input_shape=xvects.shape)
    >>> output = classify(xvects)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.ReLU,
        lin_blocks=2,
        lin_neurons=[512, 256],
        out_neurons=10,
    ):
        super().__init__(input_shape=input_shape)

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")


        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons[block_index],
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(Dropout(p=0.5), layer_name="dropout")
            self.DNN[block_name].append(activation(), layer_name="act")


        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        )

class AudioNetFeatures(nn.Module):
    def __init__(self, in_channels, conv_kernel_size, conv_dilation, out_channels, activation=nn.ReLU):
        super().__init__()
        self.conv = Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=conv_kernel_size,
                        dilation=conv_dilation,
                    )
        self.activ = activation()
        self.pool = Pooling1d(kernel_size=3, pool_type="max", stride=2)


    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        x = self.pool(x)
        return x