import sys, os, time, uuid, random, pickle, argparse, collections, csv, torch, torch.nn as nn, numpy as np
import torch.optim as optim, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from argparse import Namespace

sys.path.append("../")

from builder_actions_data_loader import BuilderActionsDataset
from encoder_utterances.model import EncoderRNN
from decoder.model import ActionsDecoder
from utils import *
from vocab import Vocabulary
import generate
from decoding import initialize_with_context

def main(args):
    """ Trains one model given the specified arguments. """

    # rng stuff
    initialize_rngs(args.seed, torch.cuda.is_available())

    # checkpoint time
    start_time = time.time()

    # create a (unique) new directory for this model based on timestamp
    model_path = os.path.join(args.model_path, args.model)
    date_dir = args.date_dir
    timestamp_dir = str(int(round(start_time*1000)))
    model_path = os.path.join(model_path, date_dir, timestamp_dir)

    if not args.suppress_logs:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:  # race condition: another model directory at this timestamp already exists, so append a random uuid and try again
            temp_path = model_path
            while os.path.exists(temp_path):
                uuid_rand = str(uuid.uuid4())
                temp_path = model_path+"-"+uuid_rand

            model_path = temp_path
            os.makedirs(model_path)

    log_path = os.path.join(model_path, args.model+'_train.log') if not args.suppress_logs else os.devnull
    sys.stdout = Logger(log_path)

    print(timestamp(), args, '\n')
    print(timestamp(), "Models will be written to", print_dir(model_path, 5))
    print(timestamp(), "Logs will be written to", print_dir(log_path, 6))

    # rng stuff
    initialize_rngs(args.seed, torch.cuda.is_available())

    # load the vocabulary
    if args.use_builder_actions and 'builder_actions' not in args.encoder_vocab_path:
        print("Error: you specified to use builder action tokens in the dialogue history, but they do not exist in the encoder's vocabulary.")
        sys.exit(0)

    if not args.use_builder_actions and 'builder_actions' in args.encoder_vocab_path:
        print("Warning: you specified not to use builder action tokens, but your encoder vocabulary contained them; resetting vocabulary to default: ../../vocabulary/glove.42B.300d-lower-1r-speaker-oov_as_unk-all_splits/vocab.pkl")
        args.encoder_vocab_path = '../../vocabulary/glove.42B.300d-lower-1r-speaker-oov_as_unk-all_splits/vocab.pkl'

    with open(args.encoder_vocab_path, 'rb') as f:
        print(timestamp(), "Loading encoder vocabulary from", print_dir(args.encoder_vocab_path, 3), "...")
        encoder_vocab = pickle.load(f)
        print(timestamp(), "Successfully loaded encoder vocabulary.\n")

    # load train and validation data
    print(timestamp(), "Loading the data (items/item batches only)...\n")

    traindataset = BuilderActionsDataset(
        itemize_args=args,

        split="train",
        saved_dataset_dir=args.saved_dataset_dir,
        load_dataset=True,
        encoder_vocab=encoder_vocab,

        items_only=True,
        load_items=False if not args.load_train_items else True,
        dump_items=False,
        development_mode=args.development_mode
    )

    train_items = traindataset.items
    collate_fn = traindataset.collate_fn

    testdataset = BuilderActionsDataset(
        itemize_args=args,

        split="val",
        saved_dataset_dir=args.saved_dataset_dir,
        load_dataset=True,
        encoder_vocab=encoder_vocab,

        items_only=True,
        load_items=False,
        dump_items=False,
        development_mode=args.development_mode
    )
    test_item_batches = testdataset.item_batches

    print(timestamp(), "Successfully loaded the data.\n")

    # write the configuration arguments to a config file in the model directory
    if not args.suppress_logs:
        with open(os.path.join(model_path, "config.txt"), "w") as f:
            args_dict = vars(args)
            for param in args_dict:
                f.write(param.ljust(20)+"\t"+str(args_dict[param])+"\n")

    print(timestamp(), "Hyperparameter configuration written to", print_dir(os.path.join(model_path, "config.txt"), 6), "\n")

    # checkpoint time
    time_post_data_preprocessing = time.time()

    # initialize the model

    # rng stuff
    initialize_rngs(args.seed, torch.cuda.is_available())

    print(timestamp(), "Initializing the model ...\n")

    """ IMPLEMENT ME FOR NEW MODELS """
    encoder = EncoderRNN(
        encoder_vocab, args.rnn_hidden_size, args.num_encoder_hidden_layers, dropout=args.dropout,
        linear_size=None, nonlinearity=None, rnn=args.rnn, bidirectional=args.bidirectional,
        train_embeddings=args.train_embeddings
    )

    if args.model == 'ActionsDecoder':
        decoder = ActionsDecoder(
            args.rnn, args.rnn_hidden_size, args.num_decoder_hidden_layers,
            args.add_posterior, args.two_dim_posterior, args.include_empty_channel, args.neighborhood_window_size,
            args.add_action_history_weight, args.concatenate_action_history_weight,
            args.num_conv_layers, args.num_unit_conv_layers, args.num_out_channels_for_init_cnn,
            args.dropout,
            args.num_linear_layers_stop_action, args.num_linear_layers_action_embedding,
            args.add_hidden_for_stop_action_pred, args.add_perspective_coords
        )
    else:
        print("Error: you have specified model", args.model, "but did not instantiate the appropriate Torch module for the model.\nPlease implement this and try again.")
        sys.exit(0)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    def print_net_info(net):
        print(net, '\n')
        print("\nNum trainable params:")

        num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        print(num_trainable_params)
        print("\n")

        return num_trainable_params

    num_trainable_params_encoder = print_net_info(encoder)
    num_trainable_params_decoder = print_net_info(decoder)

    num_trainable_params = num_trainable_params_encoder + num_trainable_params_decoder

    print("\nNum total trainable params:")
    print(num_trainable_params)
    print("\n")

    # write the configuration arguments to a config file in the model directory
    if not args.suppress_logs:
        with open(os.path.join(model_path, "config.txt"), "a") as f:
            f.write("num_trainable_params".ljust(20)+"\t"+str(num_trainable_params)+"\n")
            f.write("num_trainable_params_encoder".ljust(20)+"\t"+str(num_trainable_params_encoder)+"\n")
            f.write("num_trainable_params_decoder".ljust(20)+"\t"+str(num_trainable_params_decoder)+"\n")

    print(timestamp(), "Num trainable params added to hyperparameter configuration at", print_dir(os.path.join(model_path, "config.txt"), 6), "\n")

    print("Net parameters:")
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            print("  ", name.ljust(30), param.data.size())
    print()

    for name, param in decoder.named_parameters():
        if param.requires_grad:
            print("  ", name.ljust(30), param.data.size())
    print()

    criterion = nn.CrossEntropyLoss(size_average=False) # TODO: check
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(filter(lambda p: p.requires_grad, decoder.parameters())),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(filter(lambda p: p.requires_grad, decoder.parameters())),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    best_epoch, best_eval_result, best_validation_loss = None, None, None
    final_epoch, final_eval_result, final_validation_loss = None, None, None

    increasing = 0     # number of epochs for which validation loss has steadily increased wrt the global minimum

    all_train_losses = []
    all_val_losses = []

    if args.generation_during_training:
        # all_train_evals = []
        all_val_evals = []

        best_epoch_by_val_eval, best_val_eval = None, None
        decreasing_by_val_eval = 0

    print(timestamp(), 'Training the model for a maximum of', args.num_epochs, 'epochs.')
    if args.stop_after_n > 0:
        print(timestamp(), 'Model training will be stopped early if validation loss/f1 increases/decreases wrt the best continuously for', args.stop_after_n, 'epochs.')

    print('\n'+timestamp(), "Training the model ...\n")

    def decode(decoder, grid_repr_inputs, action_repr_inputs, labels, encoder_context, criterion): # TODO: check a little
        # print(grid_repr_inputs.shape)
        # print(action_repr_inputs.shape)
        # sys.exit(0)
        # target_inputs = to_var(decoder_inputs.target_inputs)
        # target_outputs = to_var(decoder_outputs.target_outputs)
        # TODO: check if to_var is needed
        decoder_hidden = encoder_context.decoder_hidden

        loss = 0.0
        # print(action_repr_inputs.shape)
        # print(len(action_repr_inputs[0]))
        # sys.exit(0)
        for t in range(len(action_repr_inputs[0])):
            # one time step
            action_repr_input = action_repr_inputs[0][t].view(1, 1, -1) # Next input is current target
            # print(action_repr_input.shape)
            grid_repr_input = grid_repr_inputs[0][t].unsqueeze(0)
            # print(grid_repr_input.shape)
            # decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_context)
            decoder_output, decoder_hidden, _ = decoder(
                input_seq=action_repr_input, last_hidden=decoder_hidden, input_vecs=grid_repr_input,
                posterior_dists_per_cell=None, initial_grid_repr_input=grid_repr_inputs[0][0].unsqueeze(0)
            )

            # print(labels[0][t].view(1))
            # print(labels.shape)
            # print(decoder_output.shape)
            # print(criterion(decoder_output, labels[0][t].view(1)))
            # print(loss)
            loss += criterion(decoder_output, labels[0][t].view(1))
            # print(loss)

        return loss

    try:
        # per epoch
        for epoch in range(1, args.num_epochs+1):
            if args.development_mode and epoch == 3:
                break

            print("Epoch " + str(epoch))
            epoch_start_time = time.time()

            random.shuffle(train_items) # NOTE: THIS IS MUTATING!

            # batching
            train_item_batches = []
            for i in range(0, len(train_items), args.batch_size): # 0, 32, 64, ...
                collated_batch = collate_fn(
                    train_items[i:i+args.batch_size]
                )

                train_item_batches.append(collated_batch)

            print("Training...")
            encoder.train()
            decoder.train()
            running_loss = 0.0
            for i, data in enumerate(train_item_batches, 0):
                if args.development_mode and i == 10:
                    break

                # get the inputs; data is a list of [inputs, labels]
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, _ = data
                labels = labels.long()

                if torch.cuda.is_available():
                    # prev_utterances = prev_utterances.cuda() # TODO: resolve this difference in cuda usage for encoder inputs
                    grid_repr_inputs = grid_repr_inputs.cuda()
                    action_repr_inputs = action_repr_inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                encoder_context = encoder(encoder_inputs)
                encoder_context = initialize_with_context(encoder, decoder, encoder_context, args) # TODO: check f within it

                loss = decode(decoder, grid_repr_inputs, action_repr_inputs, labels, encoder_context, criterion)

                normalized_loss = loss / float(len(labels[0]))
                normalized_loss.backward()
                # TODO: gradient clipping -- all okay?
                # TODO: which loss to include in final sum -- normalized or otherwise?

                rnn_encoder_modules = list(
                    filter(lambda x: isinstance(x, nn.GRU) or isinstance(x, nn.LSTM), list(encoder.modules()))
                )
                if rnn_encoder_modules: # clip only when there is an encoder AND RNNs exists in encoder
                    for rnn_encoder_module in rnn_encoder_modules:
                        torch.nn.utils.clip_grad_norm_(rnn_encoder_module.parameters(), args.clip)

                rnn_decoder_modules = list(
                    filter(lambda x: isinstance(x, nn.GRU) or isinstance(x, nn.LSTM), list(decoder.modules()))
                )
                if rnn_decoder_modules: # clip only when there is an decoder AND RNNs exists in decoder
                    for rnn_decoder_module in rnn_decoder_modules:
                        torch.nn.utils.clip_grad_norm_(rnn_decoder_module.parameters(), args.clip)

                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch, i + 1, running_loss / 20))
                    running_loss = 0.0

            print("Eval loss on train...")
            encoder.eval()
            decoder.eval()
            running_loss_train = 0.0
            with torch.no_grad():
                for i, data in enumerate(train_item_batches, 0):
                    if args.development_mode and i == 5:
                        break
                    # get the inputs; data is a list of [inputs, labels]
                    encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, _ = data
                    labels = labels.long()

                    if torch.cuda.is_available():
                        # prev_utterances = prev_utterances.cuda()
                        grid_repr_inputs = grid_repr_inputs.cuda()
                        action_repr_inputs = action_repr_inputs.cuda()
                        labels = labels.cuda()

                    # forward
                    encoder_context = encoder(encoder_inputs)
                    encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)

                    loss = decode(decoder, grid_repr_inputs, action_repr_inputs, labels, encoder_context, criterion)

                    # print statistics
                    running_loss_train += loss.item()

            train_result = running_loss_train
            train_result = train_result/len(train_items) # normalize

            print("Eval loss on val...")
            encoder.eval()
            decoder.eval()
            running_loss_val = 0.0
            with torch.no_grad():
                for i, data in enumerate(test_item_batches, 0):
                    if args.development_mode and i == 5:
                        break
                    # get the inputs; data is a list of [inputs, labels]
                    encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, _ = data
                    labels = labels.long()

                    if torch.cuda.is_available():
                        # prev_utterances = prev_utterances.cuda()
                        grid_repr_inputs = grid_repr_inputs.cuda()
                        action_repr_inputs = action_repr_inputs.cuda()
                        labels = labels.cuda()

                    # forward
                    encoder_context = encoder(encoder_inputs)
                    encoder_context = initialize_with_context(encoder, decoder, encoder_context, args)

                    loss = decode(decoder, grid_repr_inputs, action_repr_inputs, labels, encoder_context, criterion)

                    # print statistics
                    running_loss_val += loss.item()

            eval_result = running_loss_val
            eval_result = eval_result/len(test_item_batches) # FIXME: only works for batch size 1
            validation_loss = eval_result

            print('-'*89)
            print(timestamp(), 'End of epoch %d | Time elapsed: %5.2fs' %(epoch, time.time()-epoch_start_time))
            print(timestamp(), 'Training stats |', train_result)
            print(timestamp(), 'Validation stats |', eval_result)
            all_train_losses.append(train_result)
            all_val_losses.append(eval_result)

            # save the model per n epochs
            if args.save_per_n_epochs > 0 and epoch % args.save_per_n_epochs == 0:
                print(timestamp(), 'Saving model at epoch %d to %s ...' %(epoch, print_dir(os.path.join(model_path, args.model+'-(encoder/decoder)-epoch-%d.pkl' %(epoch)), 6)))

                if not args.suppress_logs:
                    torch.save(encoder, os.path.join(model_path, args.model+'-encoder-epoch-%d.pkl' %(epoch)))
                    torch.save(decoder, os.path.join(model_path, args.model+'-decoder-epoch-%d.pkl' %(epoch)))

            # record if this validation loss was best seen so far over epochs
            if not best_validation_loss or validation_loss <= best_validation_loss:
                print(timestamp(), 'Best model so far found at epoch %d.' %(epoch))

                if not args.suppress_logs:
                    torch.save(encoder, os.path.join(model_path, args.model+'-encoder-best_by_loss.pkl'))
                    torch.save(decoder, os.path.join(model_path, args.model+'-decoder-best_by_loss.pkl'))

                best_validation_loss = validation_loss
                best_eval_result = eval_result
                best_epoch = epoch
                increasing = 0
            else:
                increasing += 1

            if not args.suppress_logs:
                torch.save(encoder, os.path.join(model_path, args.model+'-encoder-final.pkl'))
                torch.save(decoder, os.path.join(model_path, args.model+'-decoder-final.pkl'))

            final_epoch, final_eval_result, final_validation_loss = epoch, eval_result, validation_loss

            print(timestamp(), 'Validation loss has increased wrt the best for the last', str(increasing), 'epoch(s).')

            if args.generation_during_training:
                # generation
                generation_args = Namespace(
                    model_dir=model_path,
                    saved_dataset_dir="../../../cwc-minecraft-models/data/saved_cwc_datasets/lower-builder_actions_only-no_diff-no_perspective_coords",
                    output_path=None,
                    num_workers=0,
                    seed=1234,
                    beam_size=args.beam_size,
                    max_decoding_length=args.max_decoding_length,
                    masked_decoding=args.masked_decoding,
                    development_mode=args.development_mode_generation,
                    regenerate_sentences=args.regenerate_sentences,
                    model_iteration=str(epoch),
                    split=args.split_generation,
                    disable_shuffle=True,
                    eval_only=False,
                    compute_extra_stats=False,
                    init_rngs=False
                )

                # generate.main(generation_args, test_item_batches)
                action_precision, action_recall, action_f1 = generate.main(
                    generation_args,
                    test_item_batches=test_item_batches,
                    testdataset=testdataset,
                    encoder_vocab=encoder_vocab
                )

                all_val_evals.append((action_precision, action_recall, action_f1))

                # # generation
                # generation_args = Namespace(
                #     model_dir=model_path,
                #     data_dir='../../data/logs/',
                #     gold_configs_dir='../../data/gold-configurations/',
                #     saved_dataset_dir="../../../cwc-minecraft-models/data/saved_cwc_datasets/lower-builder_actions_only-no_diff-no_perspective_coords",
                #     vocab_dir="../../vocabulary/",
                #     output_path=None,
                #     num_workers=0,
                #     seed=1234,
                #     beam_size=args.beam_size,
                #     max_decoding_length=args.max_decoding_length,
                #     masked_decoding=args.masked_decoding,
                #     development_mode=args.development_mode_generation,
                #     regenerate_sentences=args.regenerate_sentences,
                #     model_iteration=str(epoch),
                #     split="train",
                #     disable_shuffle=True,
                #     eval_only=False,
                #     compute_extra_stats=False,
                #     init_rngs=False
                # )
                #
                # # generate.main(generation_args, test_item_batches)
                # action_precision, action_recall, action_f1 = generate.main(
                #     generation_args,
                #     test_item_batches=train_item_batches,
                #     testdataset=traindataset,
                #     encoder_vocab=encoder_vocab
                # )
                #
                # all_train_evals.append((action_precision, action_recall, action_f1))

                if not best_val_eval or action_f1 >= best_val_eval:
                    print(timestamp(), 'Best model so far acc to action f1 found at epoch %d.' %(epoch))

                    if not args.suppress_logs:
                        torch.save(encoder, os.path.join(model_path, args.model+'-encoder-best_by_f1.pkl'))
                        torch.save(decoder, os.path.join(model_path, args.model+'-decoder-best_by_f1.pkl'))

                    best_val_eval = action_f1
                    best_epoch_by_val_eval = epoch
                    decreasing_by_val_eval = 0
                else:
                    decreasing_by_val_eval += 1

                print(timestamp(), 'Validation action f1 has decreased wrt the best for the last', str(decreasing_by_val_eval), 'epoch(s).')

            if args.generation_during_training:
                # stop early if validation f1 has steadly decreased for too many epochs
                if args.stop_after_n > 0 and decreasing_by_val_eval >= args.stop_after_n:
                    print(timestamp(), 'Action f1 has decreased wrt the best for the last', str(args.stop_after_n), 'epochs; quitting early.')
                    raise KeyboardInterrupt
            else:
                # stop early if validation loss has steadly increased for too many epochs
                if args.stop_after_n > 0 and increasing >= args.stop_after_n:
                    print(timestamp(), 'Validation loss has increased wrt the best for the last', str(args.stop_after_n), 'epochs; quitting early.')
                    raise KeyboardInterrupt

            print('-'*89)

    except KeyboardInterrupt:  # exit gracefully if ctrl-C is used to stop training early
        print('-'*89)
        print(timestamp(), 'Exiting from training early...')
        time.sleep(0.1)

    print(timestamp(), 'Done!')

    # generate train/val losses plot
    plot_file = log_path[:-4] + ".train_val_losses.png"

    plt.plot(np.arange(1, len(all_train_losses) + 1, step=1), all_train_losses, label='train')
    plt.plot(np.arange(1, len(all_val_losses) + 1, step=1), all_val_losses, label='val')
    plt.xticks(np.arange(1, len(all_val_losses) + 1, step=1))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(plot_file)
    plt.gcf().clear()
    print("DONE WRITING " + plot_file)

    # store losses data
    pickle.dump(all_train_losses, open(log_path[:-4] + ".train_losses.pkl", "wb"))
    pickle.dump(all_val_losses, open(log_path[:-4] + ".val_losses.pkl", "wb"))
    print("DONE WRITING PICKLED LOSSES")

    if args.generation_during_training:
        if all_val_evals:
            print(timestamp(), ' Best model by eval (action f1) was found at epoch %d' %(best_epoch_by_val_eval), ' ('+str(best_val_eval)+').', sep='')

            # write evaluation stats to eval file in model directory
            if not args.suppress_logs:
                with open(os.path.join(model_path, "eval-best-action_f1.txt"), "w") as f:
                    # f.write("Best model found at epoch %d.\n" %(best_epoch))
                    # f.write(str(best_eval_result))
                    f.write("epoch " + str(best_epoch_by_val_eval))
                    f.write("\n")
                    f.write("action_f1 " + str(best_val_eval))

        # generate train/val evals plot
        plot_file = log_path[:-4] + ".train_val_evals.png"

        # plt.plot(np.arange(1, len(all_train_evals) + 1, step=1), list(map(lambda x: x[2], all_train_evals)), label='train')
        plt.plot(np.arange(1, len(all_val_evals) + 1, step=1), list(map(lambda x: x[2], all_val_evals)), label='val')
        plt.xticks(np.arange(1, len(all_val_evals) + 1, step=1))
        plt.xlabel('epoch')
        plt.ylabel('action f1')
        plt.legend()
        plt.savefig(plot_file)
        plt.gcf().clear()
        print("DONE WRITING " + plot_file)

        # store evals data
        # pickle.dump(all_train_evals, open(log_path[:-4] + ".train_evals.pkl", "wb"))
        pickle.dump(all_val_evals, open(log_path[:-4] + ".val_evals.pkl", "wb"))
        print("DONE WRITING PICKLED EVALS")

    # print stats about best overall model found and save model accordingly
    if best_validation_loss:
        print(timestamp(), ' Best model was found at epoch %d' %(best_epoch), ' ('+str(best_eval_result)+').', sep='')

        # write evaluation stats to eval file in model directory
        if not args.suppress_logs:
            with open(os.path.join(model_path, "eval-best.txt"), "w") as f:
                # f.write("Best model found at epoch %d.\n" %(best_epoch))
                # f.write(str(best_eval_result))
                f.write("epoch " + str(best_epoch))
                f.write("\n")
                f.write("validation_loss " + str(best_eval_result))

    if final_validation_loss:
        print(timestamp(), ' Final model at end of training epoch %d' %(final_epoch), ' ('+str(final_eval_result)+').', sep='')

        # write evaluation stats to eval file in model directory
        if not args.suppress_logs:
            with open(os.path.join(model_path, "eval-final.txt"), "w") as f:
                # f.write("Final model found at epoch %d.\n" %(final_epoch))
                # f.write(str(final_eval_result))
                f.write("epoch " + str(final_epoch))
                f.write("\n")
                f.write("validation_loss " + str(final_eval_result))

    print(timestamp(), "Wrote log to:", print_dir(log_path, 6))

    # checkpoint time
    time_post_training = time.time()

    # generation
    generation_args = Namespace(
        model_dir=model_path,
        saved_dataset_dir="../../../cwc-minecraft-models/data/saved_cwc_datasets/lower-builder_actions_only-no_diff-no_perspective_coords",
        output_path=None,
        num_workers=0,
        seed=1234,
        beam_size=args.beam_size,
        max_decoding_length=args.max_decoding_length,
        masked_decoding=args.masked_decoding,
        development_mode=args.development_mode_generation,
        regenerate_sentences=args.regenerate_sentences,
        model_iteration='best_by_f1' if args.generation_during_training else 'best_by_loss',
        split=args.split_generation,
        disable_shuffle=args.disable_shuffle_generation,
        eval_only=False,
        compute_extra_stats=True,
        init_rngs=True
    )

    # generate.main(generation_args, test_item_batches)
    generate.main(
        generation_args,
        test_item_batches=test_item_batches,
        testdataset=testdataset,
        encoder_vocab=encoder_vocab
    )

    # checkpoint time
    time_post_generation = time.time()

    # compute overall time elapsed
    print(timestamp())

    def print_time_elapsed(time_elapsed, purpose):
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print("Running time stats for", purpose)
        print(" Total time elapsed: %d:%02d:%02d (%.2fs)" %(h, m, s, time_elapsed), sep="")
        print("="*89,"\n")

    print_time_elapsed(time_post_data_preprocessing - start_time, "data preprocessing")
    print_time_elapsed(time_post_training - time_post_data_preprocessing, "training")
    print_time_elapsed(time_post_generation-time_post_training, "generation")

    print("\nSaving git commit hashes ...\n")
    write_commit_hashes("../..", model_path)

    sys.stdout = sys.__stdout__

    return log_path
