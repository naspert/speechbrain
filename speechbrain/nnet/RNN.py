"""Library implementing recurrent neural networks.

Author
    Mirco Ravanelli 2020, Ju-Chieh Chou 2020, Jianyuan Zhong 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class RNN(torch.nn.Module):
    """ This function implements a vanilla RNN.

    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).

    Arguments
    ---------
    hidden_size: int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
    nonlinearity: str
         Type of nonlinearity (tanh, relu).
    num_layers: int
         Number of layers to employ in the RNN architecture.
    bias: bool
        If True, the additive bias b is adopted.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    orthogonal_init: bool:
        It True, orthogonal initialization is used for the recurrent weights.
    return_hidden: bool:
        It True, the function returns the last hidden layer.
    bidirectional: bool
         if True, a bidirectioal model that scans the sequence both
         right-to-left and left-to-right is used.
.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        nonlinearity="relu",
        num_layers=1,
        bias=True,
        dropout=0.0,
        orthogonal_init=False,
        bidirectional=False,
        return_hidden=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.orthogonal_init = orthogonal_init
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden

    def init_params(self, first_input):
        """
        Initializes the parameters of the RNN.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        if len(first_input.shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(first_input.shape[2:]))

        kwargs = {
            "input_size": self.fea_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "bias": self.bias,
            "batch_first": True,
            "nonlinearity": self.nonlinearity,
        }

        self.rnn = torch.nn.RNN(**kwargs)

        if self.orthogonal_init:
            rnn_orth_init(self.rnn)

        self.rnn.to(first_input.device)

    def forward(self, x, hx=None, init_params=False):
        """Returns the output of the vanilla RNN.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if len(x.shape) == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # Needed for multi-gpu
        self.rnn.flatten_parameters()

        # Support custom inital state
        if hx is not None:
            output, hn = self.rnn(x, hx=hx)
        else:
            output, hn = self.rnn(x)

        if self.return_hidden:
            return output, hn
        else:
            return output


class LSTM(torch.nn.Module):
    """ This function implements a basic LSTM.

    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).

    Arguments
    ---------
    hidden_size: int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
    num_layers: int
         Number of layers to employ in the RNN architecture.
    bias: bool
        If True, the additive bias b is adopted.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    orthogonal_init: bool:
        It True, orthogonal initialization is used for the recurrent weights.
    return_hidden: bool:
        It True, the function returns the last hidden layer.
    bidirectional: bool
         if True, a bidirectioal model that scans the sequence both
         right-to-left and left-to-right is used.
.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = LSTM(hidden_size=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        bias=True,
        dropout=0.0,
        orthogonal_init=False,
        bidirectional=False,
        return_hidden=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.orthogonal_init = orthogonal_init
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden

    def init_params(self, first_input):
        """
        Initializes the parameters of the LSTM.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        if len(first_input.shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(first_input.shape[2:]))

        kwargs = {
            "input_size": self.fea_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "bias": self.bias,
            "batch_first": True,
        }

        self.rnn = torch.nn.LSTM(**kwargs)

        if self.orthogonal_init:
            rnn_orth_init(self.rnn)

        self.rnn.to(first_input.device)

    def forward(self, x, hx=None, init_params=False):
        """Returns the output of the LSTM.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if len(x.shape) == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # Needed for multi-gpu
        self.rnn.flatten_parameters()

        # Support custom inital state
        if hx is not None:
            output, hn = self.rnn(x, hx=hx)
        else:
            output, hn = self.rnn(x)

        if self.return_hidden:
            return output, hn
        else:
            return output


class GRU(torch.nn.Module):
    """ This function implements a basic GRU.

    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).

    Arguments
    ---------
    hidden_size: int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
    num_layers: int
         Number of layers to employ in the RNN architecture.
    bias: bool
        If True, the additive bias b is adopted.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    orthogonal_init: bool:
        It True, orthogonal initialization is used for the recurrent weights.
    return_hidden: bool:
        It True, the function returns the last hidden layer.
    bidirectional: bool
         if True, a bidirectioal model that scans the sequence both
         right-to-left and left-to-right is used.
.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = GRU(hidden_size=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        bias=True,
        dropout=0.0,
        orthogonal_init=False,
        bidirectional=False,
        return_hidden=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.orthogonal_init = orthogonal_init
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden

    def init_params(self, first_input):
        """
        Initializes the parameters of the GRU.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        if len(first_input.shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(first_input.shape[2:]))

        kwargs = {
            "input_size": self.fea_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "bias": self.bias,
            "batch_first": True,
        }

        self.rnn = torch.nn.GRU(**kwargs)

        if self.orthogonal_init:
            rnn_orth_init(self.rnn)

        self.rnn.to(first_input.device)

    def forward(self, x, hx=None, init_params=False):
        """Returns the output of the GRU.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if len(x.shape) == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # Needed for multi-gpu
        self.rnn.flatten_parameters()

        # Support custom inital state
        if hx is not None:
            output, hn = self.rnn(x, hx=hx)
        else:
            output, hn = self.rnn(x)

        if self.return_hidden:
            return output, hn
        else:
            return output


class LiGRU(torch.nn.Module):
    """ This function implements a Light GRU (liGRU).

    Ligru is single-gate GRU model based on batch-norm + relu
    activations + recurrent dropout. For more info see:

    "M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio,
    Light Gated Recurrent Units for Speech Recognition,
    in IEEE Transactions on Emerging Topics in Computational Intelligence,
    2018" (https://arxiv.org/abs/1803.10225)

    To speed it up, it is compiled with the torch just-in-time compiler (jit)
    right before using it.

    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).

    Arguments
    ---------
    hidden_size: int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
    nonlinearity: str
         Type of nonlinearity (tanh, relu).
    normalization: str
         Type of normalization for the ligru model (batchnorm, layernorm).
         Every string different from batchnorm and layernorm will result
         in no normalization.
    num_layers: int
         Number of layers to employ in the RNN architecture.
    bias: bool
        If True, the additive bias b is adopted.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    orthogonal_init: bool:
        It True, orthogonal initialization is used for the recurrent weights.
    return_hidden: bool:
        It True, the function returns the last hidden layer.
    bidirectional: bool
         if True, a bidirectioal model that scans the sequence both
         right-to-left and left-to-right is used.
.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = LiGRU(hidden_size=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        nonlinearity="relu",
        normalization="batchnorm",
        num_layers=1,
        bias=True,
        dropout=0.0,
        orthogonal_init=False,
        bidirectional=False,
        return_hidden=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.normalization = normalization
        self.bias = bias
        self.dropout = dropout
        self.orthogonal_init = orthogonal_init
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden

    def init_params(self, first_input):
        """
        Initializes the parameters of the liGRU.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        if len(first_input.shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(first_input.shape[2:]))
        self.batch_size = first_input.shape[0]
        self.device = first_input.device

        self.rnn = self._init_layers()

        if self.orthogonal_init:
            rnn_orth_init(self.rnn)

    def _init_layers(self,):
        """
        Initializes the layers of the liGRU.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = LiGRU_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                nonlinearity=self.nonlinearity,
                normalization=self.normalization,
                bidirectional=self.bidirectional,
                device=self.device,
            ).to(self.device)
            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size
        return rnn

    def forward(self, x, hx=None, init_params=False):
        """Returns the output of the liGRU.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if len(x.shape) == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # run ligru
        output, hh = self._forward_ligru(x, hx=hx)

        if self.return_hidden:
            return output, hh
        else:
            return output

    def _forward_ligru(self, x, hx):
        """Returns the output of the vanilla liGRU.

        Arguments
        ---------
        x : torch.Tensor
        """
        h = []
        if hx is not None:
            if self.bidirectional:
                hx = hx.reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )
        # Processing the different layers
        for i, ligru_lay in enumerate(self.rnn):
            if hx is not None:
                x = ligru_lay(x, hx=hx[i])
            else:
                x = ligru_lay(x, hx=None)
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)

        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)

        return x, h


class LiGRU_Layer(torch.jit.ScriptModule):
    """ This function implements Light-Gated Recurrent Units (ligru) layer.

    Arguments
    ---------
    input_size: int
        Feature dimensionality of the input tensors.
    batch_size: int
        Batch size of the input tensors.
    hidden_size: int
         Number of output neurons .
    num_layers: int
         Number of layers to employ in the RNN architecture.
    nonlinearity: str
         Type of nonlinearity (tanh, relu).
    normalization: str
         Type of normalization (batchnorm, layernorm).
         Every string different from batchnorm and layernorm will result
         in no normalization.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    bidirectional: bool
         if True, a bidirectioal model that scans the sequence both
         right-to-left and left-to-right is used.
    device: str
         Device used for running the computations (e.g, 'cpu', 'cuda').
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="relu",
        normalization="batchnorm",
        bidirectional=False,
        device="cuda",
    ):

        super(LiGRU_Layer, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        self.w = nn.Linear(
            self.input_size, 2 * self.hidden_size, bias=False
        ).to(device)

        self.u = nn.Linear(
            self.hidden_size, 2 * self.hidden_size, bias=False
        ).to(device)

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initializing batch norm
        self.normalize = False

        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(2 * self.hidden_size, momentum=0.05).to(
                device
            )
            self.normalize = True

        elif normalization == "layernorm":
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size).to(device)
            self.normalize = True
        else:
            # Normalization is disabled here. self.norm is only  formally
            # initialized to avoid jit issues.
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size).to(device)
            self.normalize = True

        # Initial state
        self.h_init = torch.zeros(
            1, self.hidden_size, requires_grad=False, device=self.device,
        )

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop(self.batch_size)

        # Initilizing dropout
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(device)

        self.drop_mask_te = torch.tensor([1.0], device=self.device).float()

        # Setting the activation function
        if nonlinearity == "tanh":
            self.act = torch.nn.Tanh().to(device)
        else:
            self.act = torch.nn.ReLU().to(device)

    @torch.jit.script_method
    def forward(self, x, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the liGRU layer.

        Arguments
        ---------
        x : torch.Tensor
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Apply batch normalization
        if self.normalize:
            w_bn = self.norm(w.reshape(w.shape[0] * w.shape[1], w.shape[2]))
            w = w_bn.reshape(w.shape[0], w.shape[1], w.shape[2])

        # Processing time steps
        if hx is not None:
            h = self._ligru_cell(w, hx)
        else:
            h = self._ligru_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    @torch.jit.script_method
    def _ligru_cell(self, w, ht):
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        hiddens = []

        # Sampling dropout mask
        drop_mask = self._sample_drop_mask()

        # Loop over time axis
        for k in range(w.shape[1]):
            gates = w[:, k] + self.u(ht)
            at, zt = gates.chunk(2, 1)
            zt = torch.sigmoid(zt)
            hcand = self.act(at) * drop_mask
            ht = zt * ht + (1 - zt) * hcand
            hiddens.append(ht)

        # Stacking hidden states
        h = torch.stack(hiddens, dim=1)
        return h

    def _init_drop(self, batch_size):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(
            self.device
        )
        self.drop_mask_te = torch.tensor([1.0], device=self.device).float()

        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.drop_masks = self.drop(
            torch.ones(self.N_drop_masks, self.hidden_size, device=self.device,)
        ).data

    @torch.jit.script_method
    def _sample_drop_mask(self,):
        """Selects one of the pre-defined dropout masks
        """
        if self.training:

            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = (
                    self.drop(torch.ones(self.N_drop_masks, self.hidden_size,))
                    .to(self.device)
                    .data
                )

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            drop_mask = self.drop_mask_te

        return drop_mask

    @torch.jit.script_method
    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size, device=self.device,
                    )
                ).data


class QRNNLayer(torch.jit.ScriptModule):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Arguments
    ---------
    input_size : int
        The number of expected features in the input x.
    hidden_size : int
        The number of features in the hidden state h. If not specified, the input size is used.
    zoneout : float
        Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
    output_gate : bool
        If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Example
    -------
    >>> import torch
    >>> model = QRNNLayer(60, 256, bidirectional=True)
    >>> a = torch.rand([10, 120, 60])
    >>> b = model(a)
    >>> print(b[0].shape)
    torch.Size([10, 120, 512])
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional,
        zoneout=0.0,
        output_gate=True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.zoneout = zoneout
        self.output_gate = output_gate
        self.bidirectional = bidirectional

        stacked_hidden = (
            3 * self.hidden_size if self.output_gate else 2 * self.hidden_size
        )
        self.w = torch.nn.Linear(input_size, stacked_hidden, True)

        self.z_gate = nn.Tanh()
        self.f_gate = nn.Sigmoid()
        if self.output_gate:
            self.o_gate = nn.Sigmoid()

    @torch.jit.script_method
    def forgetMult(self, f, x, hidden):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        result = []
        htm1 = hidden
        hh = f * x

        for i in range(hh.shape[0]):
            h_t = hh[i, :, :]
            ft = f[i, :, :]
            if htm1 is not None:
                h_t = h_t + (1 - ft) * htm1
            result.append(h_t)
            htm1 = h_t

        return torch.stack(result)

    @torch.jit.script_method
    def split_gate_inputs(self, y):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]] # noqa F821
        if self.output_gate:
            z, f, o = y.chunk(3, dim=-1)
        else:
            z, f = y.chunk(2, dim=-1)
            o = None
        return z, f, o

    @torch.jit.script_method
    def forward(self, x, hidden=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor] # noqa F821
        """Returns the output of the QRNN layer.

        Arguments
        ---------
        x : torch.Tensor
            input to transform linearly.
        """
        if len(x.shape) == 4:
            # if input is a 4d tensor (batch, time, channel1, channel2)
            # reshape input to (batch, time, channel)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # give a tensor of shape (time, batch, channel)
        x = x.permute(1, 0, 2)
        if self.bidirectional:
            x_flipped = x.flip(0)
            x = torch.cat([x, x_flipped], dim=1)

        # note: this is equivalent to doing 1x1 convolution on the input
        y = self.w(x)

        z, f, o = self.split_gate_inputs(y)

        z = self.z_gate(z)
        f = self.f_gate(f)
        if o is not None:
            o = self.o_gate(o)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = (
                    torch.empty(f.shape)
                    .bernoulli_(1 - self.zoneout)
                    .to(f.get_device())
                ).detach()
                f = f * mask
            else:
                f = f * (1 - self.zoneout)

        z = z.contiguous()
        f = f.contiguous()

        # Forget Mult
        c = self.forgetMult(f, z, hidden)

        # Apply output gate
        if o is not None:
            h = o * c
        else:
            h = c

        # recover shape (batch, time, channel)
        c = c.permute(1, 0, 2)
        h = h.permute(1, 0, 2)

        if self.bidirectional:
            h_fwd, h_bwd = h.chunk(2, dim=0)
            h_bwd = h_bwd.flip(1)
            h = torch.cat([h_fwd, h_bwd], dim=2)

            c_fwd, c_bwd = c.chunk(2, dim=0)
            c_bwd = c_bwd.flip(1)
            c = torch.cat([c_fwd, c_bwd], dim=2)

        return h, c[-1, :, :]


class QRNN(nn.Module):
    """This is a implementation for the Quasi-RNN

    https://arxiv.org/pdf/1611.01576.pdf

    Part of the code is adapted from:

    https://github.com/salesforce/pytorch-qrnn

    Arguments
    ---------
    hidden_size :
        The number of features in the hidden state h. If not specified, the input size is used.
    num_layers :
        The number of QRNN layers to produce.
    zoneout :
        Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
    output_gate :
        If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Example
    -------
    >>> a = torch.rand([8, 120, 40])
    >>> model = QRNN(256, 4, bidirectional=True)
    >>> b = model(a, init_params=True)
    >>> print(b.shape)
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        return_hidden=False,
        **kwargs,
    ):
        assert bias is True, "Removing underlying bias is not yet supported"
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_hidden = return_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout if dropout > 0 else None
        self.kwargs = kwargs

    def init_params(self, first_input):
        input_size = first_input.shape[-1]
        if len(first_input.shape) == 4:
            input_size = first_input.shape[-1] * first_input.shape[-2]

        layers = []
        for layer in range(self.num_layers):
            layers.append(
                QRNNLayer(
                    input_size
                    if layer == 0
                    else self.hidden_size * 2
                    if self.bidirectional
                    else self.hidden_size,
                    self.hidden_size,
                    self.bidirectional,
                    **self.kwargs,
                )
            )
        self.qrnn = torch.nn.ModuleList(layers)

        # for some reason, jit module cannot handle .to("cpu")...
        device = first_input.get_device()
        if device >= 0:
            self.qrnn = self.qrnn.to(device)

        if self.dropout:
            self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, x, hidden=None, init_params=False):
        if init_params:
            self.init_params(x)

        next_hidden = []

        for i, layer in enumerate(self.qrnn):
            x, h = layer(x, None if hidden is None else hidden[i])

            next_hidden.append(h)

            if self.dropout and i < len(self.qrnn) - 1:
                x = self.dropout(x)

        next_hidden = torch.cat(next_hidden, 0).view(
            self.num_layers, *next_hidden[0].shape[-2:]
        )

        if self.return_hidden:
            return x, next_hidden
        else:
            return x


def rnn_orth_init(module):
    """
    This function is used to initialize the RNN weight with orthogonality.

    Arguments
    ---------
    module: torch.nn.Module
        Recurrent neural network module.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>> rnn_orth_init(net)
    """
    for name, param in module.named_parameters():
        if "weight_hh" in name or ".u.weight" in name:
            nn.init.orthogonal_(param)
