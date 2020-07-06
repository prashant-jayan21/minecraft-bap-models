import torch.nn as nn
import torch.nn.functional as F
import torch, sys

class WorldStateEncoderCNN(nn.Module):
    def __init__(self, add_perspective_coords, add_posterior, two_dim_posterior, include_empty_channel, neighborhood_window_size, add_action_history_weight, concatenate_action_history_weight, num_conv_layers, num_unit_conv_layers, num_out_channels_for_init_cnn, dropout, rnn_hidden_size=0):
        super(WorldStateEncoderCNN, self).__init__()

        # store model hyperparams
        self.add_perspective_coords = add_perspective_coords
        self.add_posterior = add_posterior
        self.two_dim_posterior = two_dim_posterior
        self.include_empty_channel = include_empty_channel
        self.neighborhood_window_size = neighborhood_window_size
        self.add_action_history_weight = add_action_history_weight
        self.concatenate_action_history_weight = concatenate_action_history_weight
        self.num_conv_layers = num_conv_layers
        self.num_unit_conv_layers = num_unit_conv_layers
        self.num_out_channels_for_init_cnn = num_out_channels_for_init_cnn
        self.rnn_hidden_size = rnn_hidden_size

        self.input_dim_per_neighborhood_cell = 6
        if include_empty_channel:
            self.input_dim_per_neighborhood_cell += 1
        if add_action_history_weight and concatenate_action_history_weight:
            self.input_dim_per_neighborhood_cell += 1
        if add_posterior:
            if two_dim_posterior:
                self.input_dim_per_neighborhood_cell += 2
            else:
                self.input_dim_per_neighborhood_cell += 8
        if add_perspective_coords:
            self.input_dim_per_neighborhood_cell += 3 # perspective coords

        if neighborhood_window_size == 1:
            self.num_neighborhood_cells = 27
            self.kernel_size = 3
        elif neighborhood_window_size == 2:
            self.num_neighborhood_cells = 125
            self.kernel_size = 5

        if self.num_out_channels_for_init_cnn == None:
            input_size_for_classification = int((self.input_dim_per_neighborhood_cell*self.num_neighborhood_cells)/2) # auto compute
        else:
            input_size_for_classification = self.num_out_channels_for_init_cnn

        self.conv_layers = []
        for i in range(num_conv_layers):
            if i == 0:
                layer = nn.Conv3d(
                    self.input_dim_per_neighborhood_cell,
                    input_size_for_classification,
                    kernel_size=self.kernel_size, stride=1, padding=self.neighborhood_window_size
                )
            else:
                intermediate_unit_conv_layer = nn.Conv3d(
                    input_size_for_classification,
                    int(input_size_for_classification/2),
                    kernel_size=1, stride=1, padding=0
                )
                input_size_for_classification = int(input_size_for_classification/2)

                self.conv_layers.append(intermediate_unit_conv_layer)

                layer = nn.Conv3d(
                    input_size_for_classification,
                    input_size_for_classification*2,
                    kernel_size=self.kernel_size, stride=1, padding=self.neighborhood_window_size
                )
                input_size_for_classification = input_size_for_classification*2

            self.conv_layers.append(layer)

        input_size_for_classification = input_size_for_classification + self.rnn_hidden_size

        for i in range(num_unit_conv_layers):
            if not i == num_unit_conv_layers - 1:
                layer = nn.Conv3d(
                    input_size_for_classification,
                    int(input_size_for_classification/2),
                    kernel_size=1, stride=1, padding=0
                )
                input_size_for_classification = int(input_size_for_classification/2)
            else:
                layer = nn.Conv3d(
                    input_size_for_classification,
                    7,
                    kernel_size=1, stride=1, padding=0
                )
                input_size_for_classification = 7

            self.conv_layers.append(layer)

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.nonlinearity = nn.ReLU()

        self.dropout = nn.Dropout3d(p=dropout)

    def forward(self, input_vecs, posterior_dists_per_cell, rnn_hidden_state=None):
        if self.add_posterior:
            x = torch.cat([input_vecs, posterior_dists_per_cell], dim=1)
        else:
            x = input_vecs

        for i, conv in enumerate(self.conv_layers):
            if self.rnn_hidden_size != 0 and i == ((self.num_conv_layers - 1) * 2 + 1):
                # concatenate [1, 100] hidden state with [batch_size, repr, 11, 9, 11]
                hiddens = rnn_hidden_state.repeat(1, 11, 9, 11, 1).permute(0, 4, 1, 2, 3)
                x = torch.cat([x, hiddens], dim=1)

            if i == len(self.conv_layers) - 1:
                x = conv(x) # NOTE: no final nonlinearity here -- softmax will be applied later
            else:
                x = self.dropout(self.nonlinearity(conv(x)))
                if i == len(self.conv_layers) - 2:
                    # store repr
                    penultimate_conv_layer_repr = x

        # print(penultimate_conv_layer_repr.shape) # [batch_size, 7, 11, 9, 11]
        # print(x.shape) # [batch_size, 7, 11, 9, 11]
        #
        # sys.exit(0)

        # obtaing right ordering of dimensions to correspond with ordering of labels
        penultimate_conv_layer_repr = penultimate_conv_layer_repr.permute(0, 2, 3, 4, 1) # [batch_size, 11, 9, 11, 7]

        # flatten before softmax
        penultimate_conv_layer_repr = penultimate_conv_layer_repr.reshape(-1, penultimate_conv_layer_repr.shape[4]*11*9*11)

        # obtaing right ordering of dimensions to correspond with ordering of labels
        x = x.permute(0, 2, 3, 4, 1) # [batch_size, 11, 9, 11, 7]

        # flatten before softmax
        x = x.reshape(-1, 7*11*9*11)

        return x, penultimate_conv_layer_repr # [batch_size, 7623]
