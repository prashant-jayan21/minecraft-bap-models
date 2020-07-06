import torch, sys, torch.nn as nn, re

sys.path.append('..')
from utils import *
from diff import is_feasible_next_placement

# UTILS

def initialize_with_context(encoder, decoder, encoder_context, args):
	"""
		Condition decoder on encoder's output appropriately
	"""
	def f(encoder_hidden):
		"""
			use same final encoder hidden state to initialize every decoder layer
			take encoder hidden state from something like [1, 1, 100] -> [2, 1, 100], i.e, (num_layers, batch, hidden_size)
		"""
		hidden_size = encoder_hidden.shape[2] # encoder_hidden -- [1, 1, 100]

		zeros = torch.zeros(decoder.num_hidden_layers - 1, 1, hidden_size) # [N-1, 1, 100]
		if torch.cuda.is_available():
			zeros = zeros.cuda()

		decoder_hidden = torch.cat([encoder_hidden, zeros], dim=0) # [N, 1, 100]

		return decoder_hidden

	if not args.concatenate_decoder_hidden:
		encoder_context.decoder_hidden_concat = torch.Tensor([])
		if torch.cuda.is_available():
			encoder_context.decoder_hidden_concat = encoder_context.decoder_hidden_concat.cuda()
	else:
		if not isinstance(encoder_context.decoder_hidden_concat, torch.Tensor) and not encoder_context.decoder_hidden_concat:
			encoder_context.decoder_hidden_concat = f(torch.randn(1, 1, args.decoder_hidden_concat_size))
			if torch.cuda.is_available():
				encoder_context.decoder_hidden_concat = encoder_context.decoder_hidden_concat.cuda()
		elif isinstance(encoder_context.decoder_hidden_concat, tuple):
			encoder_context.decoder_hidden_concat = encoder_context.decoder_hidden_concat[0]

	# set decoder hidden state
	if not args.set_decoder_hidden:
		encoder_context.decoder_hidden = None
	else:
		if not isinstance(encoder_context.decoder_hidden, torch.Tensor) and not encoder_context.decoder_hidden:
			print("ERROR: you specified to initialize decoder hidden state with encoder context, but no context was given.")
			sys.exit(0)

		hidden_concat = encoder_context.decoder_hidden_concat

		if isinstance(encoder_context.decoder_hidden, tuple): # true in case of lstm
			encoder_context.decoder_hidden = (
				torch.cat((encoder_context.decoder_hidden[0], hidden_concat), 2),
				torch.cat((encoder_context.decoder_hidden[1], hidden_concat), 2)
			)
		else: # true in case of gru and non-rnn modules
			encoder_context.decoder_hidden = torch.cat((encoder_context.decoder_hidden, hidden_concat), 2)

		if isinstance(encoder_context.decoder_hidden, tuple): # true in case of lstm
			encoder_context.decoder_hidden = (f(encoder_context.decoder_hidden[0]), f(encoder_context.decoder_hidden[1]))
		else: # true in case of gru and non-rnn modules
			encoder_context.decoder_hidden = f(encoder_context.decoder_hidden)

	# concatenate context to decoder inputs
	if not args.concatenate_decoder_inputs:
		encoder_context.decoder_input_concat = torch.Tensor([])
		if torch.cuda.is_available():
			encoder_context.decoder_input_concat = encoder_context.decoder_input_concat.cuda()
	else:
		if not isinstance(encoder_context.decoder_input_concat, torch.Tensor) and not encoder_context.decoder_input_concat:
			encoder_context.decoder_input_concat = torch.randn(1, 1, args.decoder_input_concat_size)
			if torch.cuda.is_available():
				encoder_context.decoder_input_concat = encoder_context.decoder_input_concat.cuda()
		elif isinstance(encoder_context.decoder_input_concat, tuple):
			encoder_context.decoder_input_concat = encoder_context.decoder_input_concat[0]

	# advance decoder by one timestep
	if args.advance_decoder_t0:
		decoder_input = encoder_context.decoder_input_t0
		_, encoder_context.decoder_hidden, _ = decoder(decoder_input, encoder_context.decoder_hidden, encoder_context, bypass_embed=True)

	return encoder_context

# BEAM SEARCH DECODING

class ActionSeq:
	def __init__(self, decoder_hidden, last_idx, built_config_post_last_action, action_history_post_last_action, seq_idxes=[], seq_scores=[], action_feasibilities=[]):
		if(len(seq_idxes) != len(seq_scores)):
			raise ValueError("length of indexes and scores should be the same")
		self.decoder_hidden = decoder_hidden
		self.last_idx = last_idx
		self.seq_idxes =  seq_idxes
		self.seq_scores = seq_scores
		self.built_config_post_last_action = built_config_post_last_action
		self.action_history_post_last_action = action_history_post_last_action
		self.action_feasibilities = action_feasibilities

	def likelihoodScore(self):
		"""
			log likelihood score
		"""
		if len(self.seq_scores) == 0:
			return -99999999.999 # TODO: check
		# return mean of sentence_score
		# TODO: Relates to the normalized loss function used when training?
		# NOTE: No need to length normalize when making selection for beam. Only needed during final selection.
		return sum(self.seq_scores) / len(self.seq_scores) # NOTE: works without rounding error because these are float tensors

	def addTopk(self, topi, topv, decoder_hidden, beam_size, EOS_tokens):
		terminates, seqs = [], []
		for i in range(beam_size):
			idxes = self.seq_idxes[:] # pass by value
			scores = self.seq_scores[:] # pass by value

			idxes.append(topi[0][i])
			scores.append(topv[0][i])

			is_feasible = is_feasible_action(self.built_config_post_last_action, topi[0][i].item())
			action_feasibilities = self.action_feasibilities[:] # pass by value
			action_feasibilities.append(is_feasible) # TODO: don't recompute feasibility in following code

			built_config_post_last_action = update_built_config(self.built_config_post_last_action, topi[0][i].item())
			action_history_post_last_action = update_action_history(self.action_history_post_last_action, topi[0][i].item(), self.built_config_post_last_action)

			seq = ActionSeq(
				decoder_hidden=decoder_hidden, last_idx=topi[0][i], built_config_post_last_action=built_config_post_last_action,
				action_history_post_last_action=action_history_post_last_action, seq_idxes=idxes, seq_scores=scores,
				action_feasibilities=action_feasibilities
			)

			if topi[0][i] in EOS_tokens:
				terminates.append((
					[idx.item() for idx in seq.seq_idxes], # TODO: need the eos token?
					seq.likelihoodScore(),
					seq.action_feasibilities,
					seq.built_config_post_last_action
				)) # tuple(word_list, score_float, action feasibilities, end_built_config)
			else:
				seqs.append(seq)

		return terminates, seqs # NOTE: terminates can be of size 0 or 1 only

def is_feasible_action(built_config_post_last_action, new_action_label):
	# new_action_label in 0-7624
	new_action = details2struct(label2details.get(new_action_label))

	if new_action.action != None:
		if new_action.action.action_type == "placement":
			return is_feasible_next_placement(block=new_action.action.block, built_config=built_config_post_last_action, extra_check=True)
		else:
			return is_feasible_next_removal(block=new_action.action.block, built_config=built_config_post_last_action)
	else: # stop action
		return True

def get_feasibility_bool_mask(built_config):
	bool_mask = []

	for action_label in range(7*11*9*11):
		bool_mask.append(is_feasible_action(built_config, action_label))

	return bool_mask

def update_built_config(built_config_post_last_action, new_action_label): # TODO: see that logic is air tight == feasibility too
	# new_action_label in 0-7624
	new_action = details2struct(label2details.get(new_action_label))

	if new_action.action != None:
		if new_action.action.action_type == "placement":
			if is_feasible_next_placement(block=new_action.action.block, built_config=built_config_post_last_action, extra_check=True):
				# print("here")
				new_built_config = built_config_post_last_action + [new_action.action.block]
			else:
				# print("there")
				new_built_config = built_config_post_last_action
		else:
			# print(built_config_post_last_action)
			if is_feasible_next_removal(block=new_action.action.block, built_config=built_config_post_last_action):
				new_built_config = list(filter(
					lambda block: block["x"] != new_action.action.block["x"] or block["y"] != new_action.action.block["y"] or block["z"] != new_action.action.block["z"],
					built_config_post_last_action
				))
			else:
				# print("there")
				new_built_config = built_config_post_last_action
			# print(new_built_config)
			# print("\n\n")
	else: # stop action
		new_built_config = built_config_post_last_action

	return new_built_config

def update_action_history(action_history_post_last_action, new_action_label, built_config_post_last_action):
	# new_action_label in 0-7624
	new_action = details2struct(label2details.get(new_action_label))

	if new_action.action != None:
		if new_action.action.action_type == "placement":
			if is_feasible_next_placement(block=new_action.action.block, built_config=built_config_post_last_action, extra_check=True):
				new_action_history = action_history_post_last_action + [new_action.action]
			else:
				new_action_history = action_history_post_last_action
		else:
			if is_feasible_next_removal(block=new_action.action.block, built_config=built_config_post_last_action):
				new_action_history = action_history_post_last_action + [new_action.action]
			else:
				new_action_history = action_history_post_last_action
		# new_action_history = action_history_post_last_action + [new_action.action] # TODO: check use of extra parens in data loader, color can be None for removals
	else: # stop action
		# print("added stop action to action history")
		new_action_history = action_history_post_last_action + [None] # TODO: replace None?

	return new_action_history

def beam_decode_action_seq(
	decoder, grid_repr_inputs, action_repr_inputs, raw_inputs,
    encoder_context, beam_size, max_length, testdataset, num_top_seqs,
	initial_grid_repr_input, masked_decoding
):
	decoder_hidden = encoder_context.decoder_hidden

	terminal_seqs, prev_top_seqs, next_top_seqs = [], [], []
	prev_top_seqs.append(
		ActionSeq(
			decoder_hidden=decoder_hidden, last_idx=torch.tensor(-1), # start token assigned action id of -1
			built_config_post_last_action=raw_inputs.initial_prev_config_raw, # same as post SOS token
			action_history_post_last_action=raw_inputs.initial_action_history_raw,
			seq_idxes=[], seq_scores=[], action_feasibilities=[]
		)
	)

	for _ in range(max_length):
		for seq in prev_top_seqs:
			# never stop action here -- .get is actually not needed
			# print(seq.last_idx)
			action_repr_input = action_label2action_repr(seq.last_idx.item()).view(1, 1, -1) # NOTE: should be [1, 1, x]
			# print(action_repr_input.shape)

			grid_repr_input = testdataset.get_repr(
	            BuilderActionExample(
	                action=None, # only ever used for computing output label which we don't need -- so None is okay
	                built_config=None,
	                prev_config=seq.built_config_post_last_action,
	                action_history=seq.action_history_post_last_action
	            ),
	            raw_inputs.perspective_coords
	        )[0].unsqueeze(0)
			# print(grid_repr_input.shape)

			if masked_decoding:
				bool_mask_input = get_feasibility_bool_mask(seq.built_config_post_last_action)

			if torch.cuda.is_available():
				action_repr_input = action_repr_input.cuda()
				grid_repr_input = grid_repr_input.cuda()

			decoder_output, decoder_hidden_new, _ = decoder(
				input_seq=action_repr_input, last_hidden=seq.decoder_hidden, input_vecs=grid_repr_input,
				posterior_dists_per_cell=None, initial_grid_repr_input=initial_grid_repr_input
			)
			# print(decoder_output.shape) # [1, 7624]

			m = nn.LogSoftmax()
			decoder_output = m(decoder_output)

			if masked_decoding:
				# mask
				for index in range(len(bool_mask_input)):
					if not bool_mask_input[index]: # infeasible action
						decoder_output[0][index] = float("-inf")

			topv, topi = decoder_output.topk(beam_size) # topv : tensor([[-0.4913, -1.9879, -2.4969, -3.6227, -4.0751]])
			term, top = seq.addTopk(topi, topv, decoder_hidden_new, beam_size, [stop_action_label_tensor])
			terminal_seqs.extend(term)
			next_top_seqs.extend(top)

		next_top_seqs.sort(key=lambda s: s.likelihoodScore(), reverse=True)
		prev_top_seqs = next_top_seqs[:beam_size]
		next_top_seqs = []

	terminal_seqs += [
		([idx.item() for idx in seq.seq_idxes], seq.likelihoodScore(), seq.action_feasibilities, seq.built_config_post_last_action) for seq in prev_top_seqs
	]
	terminal_seqs.sort(key=lambda x: x[1], reverse=True)

	# print(terminal_seqs)

	if num_top_seqs is not None:
		top_terminal_seqs = list(map(lambda x: (prune_seq(x[0], should_prune_seq(x[0])), prune_seq(x[2], should_prune_seq(x[0])), x[3]), terminal_seqs[:num_top_seqs]))
	else:
		top_terminal_seqs = list(map(lambda x: (prune_seq(x[0], should_prune_seq(x[0])), prune_seq(x[2], should_prune_seq(x[0])), x[3]), terminal_seqs))

	return top_terminal_seqs # terminal_seqs[0][0][:-1]

def generate_action_pred_seq(encoder, decoder, test_item_batches, beam_size, max_length, args, testdataset, development_mode=False, masked_decoding=False):
	encoder.eval()
	decoder.eval()

	generated_seqs, to_print = [], []
	total_examples = str(len(test_item_batches)) if not development_mode else '100'

	try:
		with torch.no_grad():
			for i, data in enumerate(test_item_batches, 0):
				if development_mode and i == 100:
					break

				# get the inputs; data is a list of [inputs, labels]
				encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, raw_inputs = data
				labels = labels.long()

				if torch.cuda.is_available(): # TODO: remove cuda here?
					# prev_utterances = prev_utterances.cuda()
					grid_repr_inputs = grid_repr_inputs.cuda()
					action_repr_inputs = action_repr_inputs.cuda()
					labels = labels.cuda()

				# forward
				encoder_context = encoder(encoder_inputs)
				encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)

				generated_seq = beam_decode_action_seq(
					decoder, grid_repr_inputs, action_repr_inputs, raw_inputs,
					encoder_context, beam_size, max_length, testdataset, 1, # TODO: parameterize 1
					initial_grid_repr_input=grid_repr_inputs[0][0].unsqueeze(0),
					masked_decoding=masked_decoding
				) # list of tuples -- [(seq, feas, end_built_configs)]

				# list(map(lambda x: x[0], generated_seq))
				# list(map(lambda x: x[1], generated_seq))

				generated_seqs.append(
					{
						"generated_seq": list(map(lambda x: x[0], generated_seq)),
                        "ground_truth_seq": labels,
						"prev_utterances": encoder_inputs.prev_utterances,
						"action_feasibilities": list(map(lambda x: x[1], generated_seq)),
						"generated_end_built_config": list(map(lambda x: x[2], generated_seq)),
						"ground_truth_end_built_config": raw_inputs.end_built_config_raw,
						"initial_built_config": raw_inputs.initial_prev_config_raw,
						"initial_action_history": raw_inputs.initial_action_history_raw
					}
				)

				if i % 20 == 0:
					print(
						timestamp(),
						'['+str(i)+'/'+total_examples+']',
						list(map(
							lambda x: ", ".join(list(map(lambda y: str(y), x))),
							list(map(lambda x: x[0], generated_seq))
						))
					)

				to_print.append(
					list(map(
						lambda x: ", ".join(list(map(lambda y: str(y), x))),
						list(map(lambda x: x[0], generated_seq))
					))
				)
	except KeyboardInterrupt:
		print("Generation ended early; quitting.")

	return generated_seqs, to_print
