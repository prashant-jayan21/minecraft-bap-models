import torch.nn as nn
import torch.nn.functional as F
import torch, sys

from encoder_world_state.model import WorldStateEncoderCNN
from utils import *

class ActionsDecoder(nn.Module):
    def __init__(
        self, rnn, rnn_hidden_size, num_hidden_layers,
        add_posterior, two_dim_posterior, include_empty_channel, neighborhood_window_size,
        add_action_history_weight, concatenate_action_history_weight, num_conv_layers, num_unit_conv_layers,
        num_out_channels_for_init_cnn, dropout,
        num_linear_layers_stop_action, num_linear_layers_action_embedding,
        add_hidden_for_stop_action_pred, add_perspective_coords
    ):
        super(ActionsDecoder, self).__init__()

        self.add_hidden_for_stop_action_pred = add_hidden_for_stop_action_pred
        self.num_hidden_layers = num_hidden_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.action_embedding_linears = []
        for i in range(num_linear_layers_action_embedding):
            layer = nn.Linear(
                11 - i,
                11 - i - 1
            )
            self.action_embedding_linears.append(layer)
        self.action_embedding_linears = nn.ModuleList(self.action_embedding_linears)

        self.rnn_input_size = 11 - num_linear_layers_action_embedding

        if rnn == "gru":
            self.rnn = nn.GRU(
                self.rnn_input_size, rnn_hidden_size, num_hidden_layers,
                dropout=(0 if num_hidden_layers == 1 else dropout), bidirectional=False,
                batch_first=True
            )
        elif rnn == "lstm":
            self.rnn = nn.LSTM(
                self.rnn_input_size, rnn_hidden_size, num_hidden_layers,
                dropout=(0 if num_hidden_layers == 1 else dropout), bidirectional=False,
                batch_first=True
            )

        self.action_pred_linear = WorldStateEncoderCNN(
            add_perspective_coords,
            add_posterior, two_dim_posterior, include_empty_channel, neighborhood_window_size,
            add_action_history_weight, concatenate_action_history_weight,
            num_conv_layers, num_unit_conv_layers, num_out_channels_for_init_cnn,
            dropout, rnn_hidden_size
        )

        penultimate_conv_layer = self.action_pred_linear.conv_layers[-2]
        self.penultimate_conv_layer_out_channels = penultimate_conv_layer.out_channels
        # print(self.penultimate_conv_layer_out_channels)

        self.maxpool = nn.MaxPool1d(kernel_size=self.penultimate_conv_layer_out_channels)

        self.input_size_for_linears_for_stop_action_pred = 11*9*11
        if self.add_hidden_for_stop_action_pred:
            self.input_size_for_linears_for_stop_action_pred += rnn_hidden_size

        self.linears_for_stop_action_pred = []
        for i in range(num_linear_layers_stop_action):
            if not i == num_linear_layers_stop_action - 1:
                layer = nn.Linear(
                    int(self.input_size_for_linears_for_stop_action_pred/(2**i)),
                    int(self.input_size_for_linears_for_stop_action_pred/(2**(i+1)))
                )
            else:
                layer = nn.Linear(
                    int(self.input_size_for_linears_for_stop_action_pred/(2**i)),
                    1
                )

            self.linears_for_stop_action_pred.append(layer)
        self.linears_for_stop_action_pred = nn.ModuleList(self.linears_for_stop_action_pred)

        self.dropout = nn.Dropout(p=dropout)

        self.nonlinearity = nn.ReLU()

        self.init_weights()

    def init_weights(self): # TODO: more generic way of implementing this?
        """ Initializes weights of linear layers with Xavier initialization. """
        for m in self.action_embedding_linears:
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)

        for m in self.linears_for_stop_action_pred:
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self, input_seq, last_hidden, input_vecs, posterior_dists_per_cell, initial_grid_repr_input
    ):
        # Note: we run this one step at a time

        # Get the embedding of the current action
        embedded = input_seq  # [1, 1, 11] -> [1, 1, 10] -> [1, 1, 9]
        for i, linear in enumerate(self.action_embedding_linears): # doesn't mutate input_seq
            embedded = self.dropout(self.nonlinearity(linear(embedded)))
        # print(embedded.shape) # [1, 1, repr]

        if(embedded.size(1) != 1):
            raise ValueError('Decoder input sequence length should be 1')

        # Get current hidden state from input word and last hidden state
        _, raw_hidden = self.rnn(embedded, last_hidden)

        if isinstance(self.rnn, nn.GRU):
            hidden = take_last_hidden(raw_hidden, self.num_hidden_layers, 1, 1, self.rnn_hidden_size)
        elif isinstance(self.rnn, nn.LSTM):
            hidden = take_last_hidden(raw_hidden[0], self.num_hidden_layers, 1, 1, self.rnn_hidden_size)

        # concat_output = rnn_output.squeeze(0)
        attn_weights = None

        # Finally predict next token
        # output = self.out(concat_output) #[64, output_size]
        hidden = hidden.squeeze(0) # [1, 1, 100] -> [1, 100]
        # print(hidden.shape)
        # print(input_vecs.shape)

        action_pred_output, penultimate_conv_layer_output = self.action_pred_linear(
            input_vecs, posterior_dists_per_cell,
            hidden
        ) # doesn't mutate hidden
        # print(action_pred_output.shape)

        input_for_stop_action_pred = self.maxpool(penultimate_conv_layer_output.unsqueeze(0)).squeeze(0)
        if self.add_hidden_for_stop_action_pred:
            input_for_stop_action_pred = torch.cat([input_for_stop_action_pred, hidden], dim=1)

        # print(input_for_stop_action_pred.shape)

        # Predict energy for stop symbol
        output_for_stop_action = input_for_stop_action_pred
        for i, linear in enumerate(self.linears_for_stop_action_pred): # doesn't mutate input_for_stop_action_pred
            if i == len(self.linears_for_stop_action_pred) - 1:
                output_for_stop_action = linear(output_for_stop_action)
            else:
                output_for_stop_action = self.dropout(self.nonlinearity(linear(output_for_stop_action)))

        # print(output_for_stop_action.shape)

        action_pred_output = torch.cat([action_pred_output, output_for_stop_action], dim=1)
        # print(action_pred_output.shape)

        hidden = hidden.unsqueeze(0)

        # Return final output, hidden state, and attention weights (for visualization)
        return action_pred_output, raw_hidden, attn_weights

    def flatten_parameters(self):
        self.rnn.flatten_parameters()
