""" A simple implementation of Bi-GRUs using PyTorch's JIT (TorchScript) """
# MIT License
#
# Copyright (c) 2021 R Mukesh, Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple
from torch import Tensor
import math


class JitGRUCell(jit.ScriptModule):
    """ Implementaion of GRU cell using JiT (Torchscript) """

    def __init__(self, input_size, hidden_size):
        super(JitGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh

        i_r, i_z, i_n = x_results.chunk(3, 1)
        h_r, h_z, h_n = h_results.chunk(3, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return n - torch.mul(n, z) + torch.mul(z, hidden)


class JitGRULayer(jit.ScriptModule):
    """ Implementation of GRU layer using JiT (torchscript) """

    def __init__(self, cell, input_size, hidden_size):
        super(JitGRULayer, self).__init__()
        self.cell = cell(input_size, hidden_size)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            hidden = self.cell(inputs[i], hidden)
            outputs += [hidden]

        return torch.stack(outputs)


class JitGRU(jit.ScriptModule):
    """ A GRU implementation using JiT (torchscript) """

    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'bidirectional', 'forward_rnn_layers', 'backward_rnn_layers']

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bias=True, bidirectional=False):

        super(JitGRU, self).__init__()
        
        # Parameter value of `bias=False`, `bidirectional=False` is not implemented
        assert bias
        assert bidirectional

        # The arguments of the GRU class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # Intialize the GRU cells in various layers of the GRU
        if num_layers == 1:
            self.forward_rnn_layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)])
            self.backward_rnn_layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)])

        else:
            self.forward_rnn_layers = nn.ModuleList(
                [
                    JitGRULayer(JitGRUCell, input_size, hidden_size) # forward GRUs in first layer
                ] + [
                    JitGRULayer(JitGRUCell, 2*hidden_size, hidden_size) # forward GRUs in consecutive layers
                    for _ in range(num_layers - 1)
                ])
            
            self.backward_rnn_layers = nn.ModuleList(
                [
                    JitGRULayer(JitGRUCell, input_size, hidden_size) # forward GRUs in first layer
                ] + [
                    JitGRULayer(JitGRUCell, 2*hidden_size, hidden_size) # forward GRUs in consecutive layers
                    for _ in range(num_layers - 1)
                ])


    @jit.script_method
    def forward(self, x, sequence_lengths, h=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]

        # If the input x has batch as first dimension (i.e., size(x): B x T x d) then transform x to have max sequence length as first dimension (i.e., size(x): T x B x d)
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # If custom initial hidden states are not supplied, initialize a tensor of all-zeros, of appropriate dimension as tensor of initial hidden states
        if h is None:
            # h = torch.zeros(self.num_layers, 2, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)
            h = torch.zeros(x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)

        output = x

        h_n = torch.empty((0, x.shape[1], self.hidden_size))

        # Perform the computations for each of the RNN layers
        for i_rnn_layer, (i_layer_forward_rnn_layer, i_layer_backward_rnn_layer) in enumerate(zip(self.forward_rnn_layers, self.backward_rnn_layers)):
            
            # Pass the outputs from last layer, corresponding initial hidden states through the forward RNN layer
            # forward_rnn_output = i_layer_forward_rnn_layer(output, h[i_rnn_layer][0])
            forward_rnn_output = i_layer_forward_rnn_layer(output, h)

            # Reverse the order of tokens in the outputs from the last layer, for feeding to a backward RNN layer
            # output_sequences_reversed = self.reverse_sequences_with_padding(output, sequence_lengths)
            output_sequences_reversed = torch.stack([
                torch.cat([output[:i_sequence_length, i_sequence].flip(dims=[0]), output[i_sequence_length:, i_sequence]])
                for i_sequence, i_sequence_length in enumerate(sequence_lengths)
            ], dim=1)
            
            # Pass the reverse of the output from the last layer, corresponding initial hidden states through the backward RNN layer
            # backward_rnn_layer_output = i_layer_backward_rnn_layer(output_sequences_reversed, h[i_rnn_layer][1])
            backward_rnn_layer_output = i_layer_backward_rnn_layer(output_sequences_reversed, h)

            # Reverse the embedding of tokens in the sequence in the output of backward RNN
            # backward_rnn_layer_output_sequences_reversed = self.reverse_sequences_with_padding(backward_rnn_layer_output, sequence_lengths)
            backward_rnn_layer_output_sequences_reversed = torch.stack([
                torch.cat([backward_rnn_layer_output[:i_sequence_length, i_sequence].flip(dims=[0]), backward_rnn_layer_output[i_sequence_length:, i_sequence]])
                for i_sequence, i_sequence_length in enumerate(sequence_lengths)
            ], dim=1)

            # Concat the embeddings of tokens outputs from the forward and backward RNN layers
            output = torch.cat([forward_rnn_output, backward_rnn_layer_output_sequences_reversed], dim=2)

            # Get the hidden state at the last time step from forward rnn layers for each sequence in the batch
            i_layer_forward_rnn_layer_h_n = torch.stack([
                forward_rnn_output[i_sequence_length-1, i_sequence]
                for i_sequence, i_sequence_length in enumerate(sequence_lengths)
            ], dim=0)

            # Get the hidden state at the last time step from backward rnn layers for each sequence in the batch
            i_layer_backward_rnn_layer_h_n = torch.stack([
                backward_rnn_layer_output[i_sequence_length-1, i_sequence]
                for i_sequence, i_sequence_length in enumerate(sequence_lengths)
            ], dim=0)

            # Combine hidden state at last time step from forward and backward rnn layers to form a tensor of shape (num_directions, batch_size, hidden_size)
            i_rnn_layer_hn = torch.stack([i_layer_forward_rnn_layer_h_n, i_layer_backward_rnn_layer_h_n], dim=0)

            # Concat the hidden state at last time step from forward and backward rnn layers from this layer with the tensor storing it for all layers
            h_n = torch.cat([h_n, i_rnn_layer_hn], dim=0)

        # Mask the garbage values at the end of sequences caused due to padding
        max_sequence_length = int(sequence_lengths[0])
        idxes = torch.arange(0, max_sequence_length, out=output.data.new_empty(max_sequence_length, dtype=torch.long)).unsqueeze(1).float()
        mask = (idxes < sequence_lengths.unsqueeze(0)).float()

        output = mask.unsqueeze(2) * output

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output, h_n

# ----------------------------------------------------------------------------------------------------------------------
import random

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def test_jit_bigru(
        max_sequence_length, batch_size, # input specifications
        input_size, # input specifications / GRU hyperparamters
        hidden_size, num_layers # GRU hyperparameters
    ):
    """ Test JiT BiGRU implementation against `nn.GRU` by creating GRU with given hyperparamters using random input with given specifications """

    padded_embedded_sequences_tensor = torch.randn(max_sequence_length, batch_size, input_size)

    ## The first sequence has length equal to max sequence length, rest have random sequence lengths from range 1 to max sequence length
    sequence_lengths_list = sorted([max_sequence_length] + [random.randint(1, max_sequence_length) for _ in range(batch_size - 1)], reverse=True)
    sequence_lengths_tensor = torch.LongTensor(sequence_lengths_list)

    ## Pack the padded input embedded sequences tensor
    packed_padded_embedded_sequences = pack_padded_sequence(padded_embedded_sequences_tensor, sequence_lengths_tensor, batch_first=False, enforce_sorted=True)

    ## Create instances of the pytorch's in-built nn.GRU and custom jit Bi-GRU
    args = (input_size, hidden_size)
    kwargs = {'num_layers': num_layers, 'bias': True, 'batch_first': False, 'bidirectional': True}

    pytorch_gru = nn.GRU(*args, **kwargs) # pytorch's in-built GRU
    jit_gru = JitGRU(*args, **kwargs) # Custom implementation of GRU in JiT

    ## Copy the random initial weights from the pytorch GRU to JiT GRU instance
    # The name of each JitGRU parameter that we've defined in JitGRUCell
    gru_cell_param_names = ["weight_hh", "weight_ih", "bias_ih", "bias_hh"]

    for i_layer in range(num_layers):
        for i_gru_layer_direction in range(2):
            for gru_cell_param_name in gru_cell_param_names:

                # Build the name of the parameters in this layer in `nn.GRU` and `JitGRU`
                pytorch_param_name = f"{gru_cell_param_name}_l{i_layer}" + ("_reverse" if i_gru_layer_direction == 1 else "")
                jit_param_name = f"{'forward_rnn_layers' if i_gru_layer_direction == 0 else 'backward_rnn_layers'}.{i_layer}.cell.{gru_cell_param_name}"

                # Get the corresponding parameter value in `nn.GRU` and `JitGRU`
                pytorch_param = pytorch_gru.state_dict()[pytorch_param_name]
                jit_param = jit_gru.state_dict()[jit_param_name]

                # Make sure that the corresponding parameters have same shape in `nn.GRU` and `JitGRU`
                assert pytorch_param.shape == jit_param.shape

                # Copy the weights values from parameters in `nn.GRU` to corresponding parameters in `JitGRU`
                with torch.no_grad():
                    pytorch_param.copy_(jit_param)

    ## Run the same inputs through both `nn.GRU` and `JitGRU` instances
    # Pass the input through `nn.GRU` and pad the packed output
    pytorch_packed_output, pytorch_h_n = pytorch_gru(packed_padded_embedded_sequences)
    pytorch_padded_output, _ = pad_packed_sequence(pytorch_packed_output)

    # Pass the input through `JitGRU`
    jit_padded_output, jit_h_n = jit_gru(padded_embedded_sequences_tensor, sequence_lengths_tensor)

    ## Make sure the output values from pytorch GRU and jit GRU are reasonably close
    assert (jit_padded_output - pytorch_padded_output).abs().max() < 1e-5, "`output` tensor from PyTorch and JiT GRUs don't match"
    assert (jit_h_n - pytorch_h_n).abs().max() < 1e-5, "`h_n` tensor from PyTorch and JiT GRUs don't match"


if __name__ == '__main__':

    """ Test 1: GRU hyperparameters and input specifications """
    max_sequence_length = 1
    batch_size = 1

    input_size = 1
    hidden_size = 1
    num_layers = 1

    test_jit_bigru(max_sequence_length, batch_size, input_size, hidden_size, num_layers)
    print("Test case 1 passed.")

    max_sequence_length = 10
    batch_size = 8

    input_size = 5
    hidden_size = 3
    num_layers = 1

    test_jit_bigru(max_sequence_length, batch_size, input_size, hidden_size, num_layers)
    print("Test case 2 passed.")

    """ Test 2: GRU hyperparameters and input specifications """
    max_sequence_length = 10
    batch_size = 8

    input_size = 5
    hidden_size = 3
    num_layers = 3

    test_jit_bigru(max_sequence_length, batch_size, input_size, hidden_size, num_layers)
    print("Test case 3 passed.")

    max_sequence_length = 123
    batch_size = 75

    input_size = 200
    hidden_size = 50
    num_layers = 10

    test_jit_bigru(max_sequence_length, batch_size, input_size, hidden_size, num_layers)
    print("Test case 4 passed.")

    print("ALL TEST CASES PASSED.")
