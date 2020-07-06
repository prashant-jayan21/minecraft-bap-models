import sys, os, argparse, time, itertools, pickle, train
from collections import OrderedDict
from utils import timestamp, print_dir, Logger, parse_value, write_commit_hashes
from vocab import Vocabulary
from data_loader import CwCDataset, BuilderAction
from builder_actions_data_loader import BuilderActionsDataset, RawInputs, EncoderInputs

def main(args):
    # NOTE: THIS SCRIPT SHOULD BE KEPT MINIMAL -- ESPCIALLY NOTHING HERE SHOULD BE NON-DETERMINISTIC IN NATURE

    """ Training script that runs through different hyperparameter settings and trains multiple models. """
    model_path = os.path.join(args.model_path, args.model)
    timestamp_dir = str(int(round(time.time()*1000)))
    args.date_dir = args.date_dir+'/'+args.model+'_trainer-'+timestamp_dir
    model_path = os.path.join(model_path, args.date_dir)

    if not os.path.exists(model_path) and not args.suppress_logs:
        os.makedirs(model_path)

    log_path = os.path.join(model_path, args.model+'_trainer-'+timestamp_dir+'.log') if not args.suppress_logs else os.devnull
    logger = Logger(log_path)
    sys.stdout = logger

    # create all combinations of hyperparameters
    param_lists = get_param_lists(args.hyperparameter_file)
    combined = combine_params(param_lists)
    params = flatten_combined_params(args.model, param_lists, combined)

    models_trained = 0
    start_time = time.time()

    # train each model
    for i in range(len(params)):
        config = params[i]

        print(timestamp(), "Training model", str(models_trained+1), "of", len(params), "...")
        print(timestamp(), "Parameters tuned:", config)

        for param in config:
            if not hasattr(args, param):
                sys.stdout = sys.__stdout__
                print("Error: you have specified param", param, "but it does not exist as a command-line argument!\nPlease implement this and try again.")
                sys.exit(0)

            setattr(args, param, config[param])

        sys.stdout = sys.__stdout__
        training_log = train.main(args)
        models_trained += 1

        sys.stdout = logger
        print(timestamp(), "Done! Model", str(models_trained), "training log saved to", print_dir(training_log, 6), "\n")

    print(timestamp(), "Model training finished. Number of models trained:", models_trained)
    time_elapsed = time.time()-start_time
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    print(timestamp(), " Total time elapsed: %d:%02d:%02d (%.2fs)" %(h, m, s, time_elapsed), sep="")

    print("\nSaving git commit hashes ...\n")
    write_commit_hashes("../..", model_path)

    print(os.path.abspath(model_path))

    sys.stdout = sys.__stdout__

# UTILS

def flatten_combined_params(model_name, param_lists, combined):
	params = []
	for combined_tuple in combined:
		config = {}
		flattened = flatten(combined_tuple)
		for i in range(len(param_lists)):
			config[list(param_lists.keys())[i]] = flattened[i]

		""" IMPLEMENT ME FOR NEW MODELS """
		if model_name == 'seq2seq':
			if not config.get("linear_size") and config.get("nonlinearity") or config.get("linear_size") and not config.get("nonlinearity"):
				continue

		if model_name == 'cnn_3d':
			if config.get('built_diff_features') != config.get('gold_diff_features'):
				continue

		params.append(config)

	print("Hyperparameter configurations ("+str(len(params))+"):")
	for param in params:
		print("\t", param)
	print()

	return params

def get_param_lists(hyperparameter_file):
	param_lists = OrderedDict()
	with open(hyperparameter_file) as f:
		print(timestamp(), "Reading hyperparameter configuration from", print_dir(hyperparameter_file, 4), "\n")

		for line in f:
			tokens = line.split()
			param = tokens[0]
			values = []

			for value in tokens[1:]:
				values.append(parse_value(value))

			param_lists[param] = values

	print("Parameter lists:", param_lists)
	return param_lists

def combine_params(param_lists):
	combined = None
	for param in param_lists:
		if not combined:
			combined = param_lists[param]
		else:
			combined = itertools.product(combined, param_lists[param])

	return combined

def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, nargs='?', default='ActionsDecoder', help='type of model to train')
    parser.add_argument('hyperparameter_file', type=str, help='file of hyperparameter options to train models for')

    # io
    parser.add_argument('--model_path', type=str, default='../../models/', help='path for saving trained models')
    parser.add_argument('--saved_dataset_dir', type=str, default="", help='path for saved dataset to use')
    parser.add_argument('--date_dir', type=str, default=time.strftime("%Y%m%d"))

    # training options
    parser.add_argument('--optimizer', type=str, default='adamw', help='adam or adamw')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay for AdamW / L2 penalty for Adam')
    parser.add_argument('--num_epochs', type=int, default=40, help='maximum possible number of epochs')
    parser.add_argument('--save_per_n_epochs', type=int, default=1, help='save models every n epochs')
    parser.add_argument('--stop_after_n', type=int, default=10, help='stop training models after n epochs of increasing loss on the validation set')
    parser.add_argument('--log_step', type=int , default=1000, help='step size for printing log info')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--seed', type=int, default=1234, help='random seed -- we recommend sticking to the default')
    parser.add_argument("--development_mode", default=False, action="store_true", help="Whether or not to run in development mode, i.e., with less data")
    parser.add_argument('--suppress_logs', default=False, action='store_true', help='suppress log messages written to disk')

    parser.add_argument('--rnn_hidden_size', type=int , default=100, help='size of the RNN hidden state for both the utterances encoder and the actions decoder')
    parser.add_argument('--num_decoder_hidden_layers', type=int, default=1, help='Number of hidden layers in the decoder')
    parser.add_argument("--bidirectional", default=False, action="store_true", help="Whether or not to use a bidirectional utterances encoder")

    parser.add_argument("--include_empty_channel", default=True, action="store_true", help="Whether to add an empty channel in the CNN input representation")
    parser.add_argument('--neighborhood_window_size', type=int, default=1, help='size of window to consider for representing neighborhood of a cell')
    parser.add_argument("--add_action_history_weight", default=True, action="store_true", help="Whether to add an extra bit for action history weights in input repr")
    parser.add_argument('--action_history_weighting_scheme', type=str, default="step", help='type of action weighting scheme to use')
    parser.add_argument("--concatenate_action_history_weight", default=True, action="store_true", help="Whether to concatenate or incorporate into the vector representing the current block in a neighborhood cell")

    parser.add_argument("--add_posterior", default=False, action="store_true", help="Whether to add 8 dim local posterior as a feature to model")
    parser.add_argument("--two_dim_posterior", default=True, action="store_true", help="Whether to only compute 2 dim placement/removal posterior")
    parser.add_argument("--add_one_smoothing", default=True, action="store_true", help="Whether to perform add-one smoothing for prior_new or not")

    parser.add_argument('--num_conv_layers', type=int, default=3, help='number of conv layers in world state encoder (denoted by m in paper)')
    parser.add_argument('--num_out_channels_for_init_cnn', type=int, default=300, help='number of output channels for first conv layer')
    parser.add_argument('--num_unit_conv_layers', type=int, default=2, help='number of 1x1x1 conv layers in action sequence decoder (denoted by n in paper)')

    parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')

    parser.add_argument('--num_linear_layers_stop_action', type=int, default=4, help='number of linear layers between rnn hidden state and stop action probability (denoted by l in paper)')
    parser.add_argument('--num_linear_layers_action_embedding', type=int, default=2, help='number of linear layers to embed an action (denoted by j in paper)')
    parser.add_argument("--add_hidden_for_stop_action_pred", default=False, action="store_true", help="Whether to add rnn hidden state as input for stop token predictor -- we recommend sticking to the default")

    parser.add_argument('--encoder_vocab_path', type=str, default='../../vocabulary/glove.42B.300d-lower-1r-speaker-builder_actions-oov_as_unk-all_splits/vocab.pkl', help='path for encoder vocabulary wrapper')
    parser.add_argument('--num_encoder_hidden_layers', type=int, default=1, help='number of hidden layers in utterances encoder RNN')
    parser.add_argument('--rnn', type=str, default="gru", help='type of RNN -- gru or lstm')
    parser.add_argument("--train_embeddings", default=False, action="store_true", help="Whether or not to have trainable embeddings -- we recommend sticking to the default")

    parser.add_argument('--num_prev_utterances', type=int, default=3, help='number of previous utterances to use as input')
    parser.add_argument('--num_prev_utterances_by_heuristic', default=False, action='store_true', help='whether to decide number of previous utterances to use as input based on the heuristic (denoted by H2 in paper)')
    parser.add_argument('--num_prev_utterances_until_last_architect', default=False, action='store_true', help='whether to include previous utterances until last Architect utterance (denoted by H1 in paper)')
    parser.add_argument('--num_prev_utterances_until_last_action', default=False, action='store_true', help='whether to include previous utterances until the last Builder action')
    parser.add_argument('--use_builder_actions', default=False, action='store_true', help='include builder action tokens in the dialogue history (this + H2 is denoted by H3 in paper)')

    parser.add_argument('--add_perspective_coords', default=False, action='store_true', help='whether or not to include perspective coords in world state repr')

    # encoder-decoder connection parameters
    parser.add_argument('--set_decoder_hidden', default=False, action='store_true', help='sets decoder hidden state to the decoder_hidden context vector produced by encoder')
    parser.add_argument('--concatenate_decoder_inputs', default=False, action='store_true', help='enables vectors of size decoder_input_concat_size to be concatenated to decoder inputs at every timestep')
    parser.add_argument('--concatenate_decoder_hidden', default=False, action='store_true', help='enables vectors of size decoder_hidden_concat_size to be concatenated to the initial provided decoder hidden state (set_decoder_hidden must be True)')
    parser.add_argument('--decoder_input_concat_size', type=int, default=0, help='size of vector to be concatenated to decoder input at every timestep; if one is not provided by the encoder, a 0-vector of this size is concatenated')
    parser.add_argument('--decoder_hidden_concat_size', type=int, default=0, help='size of vector to be concatenated to decoder hidden state at initialization; if one is not provided by the encoder, a 0-vector of this size is concatenated')
    parser.add_argument('--advance_decoder_t0', default=False, action='store_true', help='advances the decoder at start of sequence by a timestep using the decoder_input_t0 context vector produced by encoder')

    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')

    parser.add_argument('--beam_size', type=int, default=1, help='beam size for beam search decoding')
    parser.add_argument('--max_decoding_length', type=int, default=10, help='max iterations for decoder when decoding w/o ground truth inputs')
    parser.add_argument('--masked_decoding', default=False, action='store_true', help='whether or not to use constrained decoding to mask out infeasible actions')
    parser.add_argument("--development_mode_generation", default=False, action="store_true", help="Whether or not to run generation in development mode, i.e., with less data")
    parser.add_argument('--regenerate_sentences', default=False, action='store_true', help='generate sentences for a model even if a generated sentences file already exists in its directory')
    parser.add_argument('--split_generation', default='val', help='data split from which sentences should be generated')
    parser.add_argument('--disable_shuffle_generation', default=False, action='store_true', help='disable shuffling of the data to be generated from')

    parser.add_argument('--generation_during_training', default=False, action='store_true', help='generate on train and val during training, perform early stopping wrt net action F1 instead of loss')

    parser.add_argument('--load_train_items', default=False, action='store_true', help='load train items from disk instead of generating -- not mandatory but useful for when using augmented data')

    args = parser.parse_args()
    main(args)
