import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append('..')
from utils import *

class EncoderRNN(nn.Module):
	"""
	Encoder RNN (GRU/LSTM)
	"""
	# TODO: set up for attention, multiple layers
	# FIXME: use embed_dropout!!
	def __init__(self, vocabulary, hidden_size, num_hidden_layers, dropout=0, linear_size=None, nonlinearity=None, rnn="gru", bidirectional=True, train_embeddings=False):
		super(EncoderRNN, self).__init__()
		# Keep for reference
		self.rnn_hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.dropout = dropout
		self.linear_size = linear_size
		self.bidirectional = bidirectional

		# Define layers
		self.embed = vocabulary.word_embeddings
		if train_embeddings:
			self.embed.weight.requires_grad = True

		self.linear = None
		if linear_size:
			self.linear = nn.Linear(vocabulary.embed_size, self.linear_size)

		self.nonlinearity = None
		if nonlinearity == 'relu':
			self.nonlinearity = nn.ReLU()
		elif nonlinearity == 'tanh':
			self.nonlinearity = nn.Tanh()

		self.final_embedding_size = vocabulary.embed_size if not linear_size else linear_size

		self.embed_dropout = nn.Dropout(p=(dropout if self.linear or self.embed.weight.requires_grad else 0))

		if rnn == "gru":
			self.rnn = nn.GRU(self.final_embedding_size, hidden_size, num_hidden_layers, dropout=(0 if num_hidden_layers == 1 else dropout), bidirectional=self.bidirectional, batch_first=True)
		elif rnn == "lstm":
			self.rnn = nn.LSTM(self.final_embedding_size, hidden_size, num_hidden_layers, dropout=(0 if num_hidden_layers == 1 else dropout), bidirectional=self.bidirectional, batch_first=True)

		# TODO: Generalize to avoid using magic numbers
		self.input_encoding_size = self.rnn_hidden_size # NOTE: even in bidirectional case because we sum forward and backward final hidden states

		self.init_weights()

	def init_weights(self):
		""" Initializes weights of linear layers with Xavier initialization. """
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.bias.data.zero_()
				nn.init.xavier_uniform_(m.weight)

	def forward(self, encoder_inputs):
		input_seq = to_var(encoder_inputs.prev_utterances)
		input_lengths = encoder_inputs.prev_utterances_lengths
		embedded = self.embed(input_seq)
		embedded = self.embed_dropout(embedded) #[1, 64, 512]  # [1, 1, 300]
		packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
		_, hidden = self.rnn(packed) # hidden: (num_layers * num_directions, batch, hidden_size)

		if isinstance(self.rnn, nn.GRU):
			hidden = take_last_hidden(hidden, self.num_hidden_layers, 2 if self.bidirectional else 1, 1, self.rnn_hidden_size)
		elif isinstance(self.rnn, nn.LSTM):
			hidden = (
				take_last_hidden(hidden[0], self.num_hidden_layers, 2 if self.bidirectional else 1, 1, self.rnn_hidden_size),
				take_last_hidden(hidden[1], self.num_hidden_layers, 2 if self.bidirectional else 1, 1, self.rnn_hidden_size)
			)

		if self.bidirectional:
			def f(hidden):
				"""
					sum final forward and backward hidden states
					take hidden from something like [2, 1, 100] -> [1, 1, 100]
				"""
				return torch.sum(hidden, dim=0, keepdim=True)

			if isinstance(self.rnn, nn.GRU):
				hidden = f(hidden)
			elif isinstance(self.rnn, nn.LSTM):
				hidden = (f(hidden[0]), f(hidden[1]))

		return EncoderContext(decoder_hidden=hidden, decoder_input_concat=hidden, decoder_hidden_concat=hidden, decoder_input_t0=hidden)

	def flatten_parameters(self):
		self.rnn.flatten_parameters()
