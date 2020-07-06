import os, argparse, torch, pickle, pprint, json, sys, numpy as np, random
from glob import glob
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append('..')
from utils import *
from vocab import Vocabulary
from builder_actions_data_loader import BuilderActionsDataset
from decoding import generate_action_pred_seq
from argparse import Namespace
from diff import diff, get_diff, dict_to_tuple

def main(args, test_item_batches=None, testdataset=None, encoder_vocab=None):
    if args.init_rngs:
        # rng stuff
        initialize_rngs(args.seed, torch.cuda.is_available())

    if not args.development_mode:
        torch.multiprocessing.set_sharing_strategy('file_system')

    config_file = os.path.join(args.model_dir, "config.txt")
    _, config_params = get_config_params(config_file)
    print(config_params)

    if args.eval_only:
        print("\n\nRUNNING IN EVAL ONLY MODE...\n")
    else:
        print("\n\nRUNNING IN GENERATE AND EVAL MODE...\n")

    output_path = args.output_path
    args_sfx = None

    if output_path is None:
        args_sfx = '-'+args.model_iteration+'-'+args.split+('' if not args.development_mode else '-development_mode')
        args_sfx += '-beam_'+str(args.beam_size)
        args_sfx += ('' if not args.masked_decoding else '-masked')
        output_path = os.path.join(args.model_dir, 'generated_sentences'+args_sfx+'.txt')
        output_path_generated_seqs = os.path.join(args.model_dir, 'generated_seqs_raw'+args_sfx+'-cpu.p') # TODO: what when output_path is not None?
    else:
        output_path = os.path.join(args.model_dir, output_path)
        output_path_generated_seqs = output_path.replace(".txt", "-cpu.p")
        args_sfx = ""

    if not args.regenerate_sentences and (os.path.isfile(output_path) or os.path.isfile(os.path.join(args.model_dir, 'fail.txt'))):
        print("\nGenerated sentences already exist for model", args.model_dir+"; skipping.\n")
        return

    if args.model_iteration in ["best", "final"]:
        if not args.eval_only:
            eval_files = glob(args.model_dir+'/eval-*.txt')
            if len(eval_files) == 0:
                print("Model", args.model_dir, "has not finished training; skipping.")
                return

    print("\n"+timestamp(), "Generated sentences will be written to", print_dir(output_path, 6), "...\n")

    if not args.eval_only:
        model_type = config_params["model"]

        model_files = glob(args.model_dir+"/*-"+args.model_iteration+".pkl")

        models = {}
        for model_file in model_files:
            with open(model_file, 'rb') as f:
                if not torch.cuda.is_available():
                    model = torch.load(f, map_location="cpu")
                else:
                    model = torch.load(f)
                if "flatten_parameters" in dir(model):
                    model.flatten_parameters() # TODO: check, flatten for all sub-modules recursively
                if "encoder" in model_file:
                    models["encoder"] = model
                elif "decoder" in model_file:
                    models["decoder"] = model

        saved_dataset_dir = config_params['saved_dataset_dir'] if args.saved_dataset_dir is None else args.saved_dataset_dir

        if test_item_batches == None and testdataset == None and encoder_vocab == None:
            # load the vocabulary
            with open(config_params["encoder_vocab_path"], 'rb') as f:
                encoder_vocab = pickle.load(f)

            # load test/validation data
            print(timestamp(), "Loading the data (item batches only)...\n")

            itemize_args = Namespace(**config_params)
            setattr(itemize_args, "num_workers", args.num_workers)

            testdataset = BuilderActionsDataset(
                itemize_args=itemize_args,

                split=args.split,
                saved_dataset_dir=saved_dataset_dir,
                load_dataset=True,
                encoder_vocab=encoder_vocab,

                items_only=True,
                load_items=False,
                dump_items=False,
                development_mode=args.development_mode
            )
            test_item_batches = testdataset.item_batches

            print(timestamp(), "Successfully loaded the data.\n")

        else:
            print("Using same data as during training time...")

        print("Done generating data.")

        print(models)

        init_args = Namespace(set_decoder_hidden=config_params['set_decoder_hidden'],
                          concatenate_decoder_inputs=config_params['concatenate_decoder_inputs'],
                          concatenate_decoder_hidden=config_params['concatenate_decoder_hidden'],
                          decoder_input_concat_size=config_params['decoder_input_concat_size'],
                          decoder_hidden_concat_size=config_params['decoder_hidden_concat_size'],
                          advance_decoder_t0=config_params['advance_decoder_t0'])
    else:
        # only vocab needed in eval only mode
        if test_item_batches == None and testdataset == None and encoder_vocab == None:
            with open(config_params["encoder_vocab_path"], 'rb') as f:
                encoder_vocab = pickle.load(f)
        else:
            print("Using same data as during training time...")

    try:
        to_print = None

        if not args.eval_only:
            generated_seqs, to_print = generate_action_pred_seq(
                encoder=models["encoder"], decoder=models["decoder"], test_item_batches=test_item_batches,
                beam_size=args.beam_size, max_length=args.max_decoding_length, args=init_args,
                testdataset=testdataset,
                development_mode=args.development_mode,
                masked_decoding=args.masked_decoding
            )

            # write generated seqs to disk
            print("\n"+timestamp(), "Pickling raw generated seqs to", print_dir(output_path_generated_seqs, 6), "...\n")
            # move gpu tensors to cpu for cross-platform support
            for output_obj in generated_seqs:
                output_obj['ground_truth_seq'] = output_obj['ground_truth_seq'].cpu()
            pickle.dump(generated_seqs, open(output_path_generated_seqs, "wb")) # always non-shuffled
            print("Done!\n")
        else:
            # read generated seqs from disk
            print("\n"+timestamp(), "Reading pickled raw generated seqs from", print_dir(output_path_generated_seqs, 6), "...\n")
            generated_seqs = pickle.load(open(output_path_generated_seqs, "rb")) # always non-shuffled
            print("Done!\n")

        if not args.disable_shuffle: # only dictates whether human-readable output files are shuffled or not
            random.shuffle(generated_seqs)
            if to_print != None:
                random.shuffle(to_print)

        # Eval
        def format_label_seq(label_seq):
            formatted_seq = []
            for label in label_seq:
                formatted_action = details2struct(label2details.get(label)).action
                formatted_action_dict = {
                    key: formatted_action.__dict__[key] for key in formatted_action.__dict__ if key != 'weight'
                } # TODO: see if you can avoid the skipping of the weight key

                formatted_seq.append(formatted_action_dict)

            return formatted_seq

        def format_generated_seq(generated_seq):
            generated_seq_formatted = []

            for label_seq in generated_seq:
                generated_seq_formatted.append(format_label_seq(label_seq))

            return generated_seq_formatted

        def format_ground_truth_seq(ground_truth_seq):
            ground_truth_seq = ground_truth_seq[0].tolist() #[:-1]
            ground_truth_seq = prune_seq(ground_truth_seq, should_prune_seq(ground_truth_seq))

            return format_label_seq(ground_truth_seq)

        def format_prev_utterances(prev_utterances):
            prev_utterances = list(map(lambda x: list(map(lambda y: encoder_vocab.idx2word[y.item()], x)), prev_utterances))
            prev_utterances = list(map(lambda x: " ".join(x), prev_utterances))[0]
            return prev_utterances

        def format(output_obj):
            generated_seq = format_generated_seq(output_obj["generated_seq"])
            ground_truth_seq = format_ground_truth_seq(output_obj["ground_truth_seq"])
            prev_utterances = format_prev_utterances(output_obj["prev_utterances"])
            return {
                "generated_seq": generated_seq,
                "ground_truth_seq": ground_truth_seq,
                "prev_utterances": prev_utterances,
                "action_feasibilities": output_obj["action_feasibilities"]
            }

        all_edit_distances = []
        all_mean_edit_distances_per_alignment = []
        tp, fp, fn = 0, 0, 0
        for output_obj in generated_seqs:
            initial_built_config = output_obj["initial_built_config"]
            generated_end_built_config = output_obj["generated_end_built_config"][0]
            ground_truth_end_built_config = output_obj["ground_truth_end_built_config"]

            # compare
            if args.compute_extra_stats:
                diff_dict = diff(gold_config=ground_truth_end_built_config, built_config=generated_end_built_config)

                edit_distance = len(diff_dict["gold_minus_built"]) + len(diff_dict["built_minus_gold"])
                all_edit_distances.append(edit_distance)

                # factor for multiple alignments
                if generated_end_built_config == [] and ground_truth_end_built_config == []:
                    # edit_distances_per_alignment = [0] * 11*11*4
                    mean_edit_distance_per_alignment = 0.0
                else:
                    _, perturbations_and_diffs = get_diff(gold_config=ground_truth_end_built_config, built_config=generated_end_built_config)
                    diffs_built_config_space = list(map(lambda x: x.diff.diff_built_config_space, perturbations_and_diffs))

                    edit_distances_per_alignment = list(map(lambda x: len(x["gold_minus_built"]) + len(x["built_minus_gold"]), diffs_built_config_space))

                    mean_edit_distance_per_alignment = np.mean(edit_distances_per_alignment)

                all_mean_edit_distances_per_alignment.append(mean_edit_distance_per_alignment)

            # P/R/F of actions
            net_change_generated = diff(gold_config=generated_end_built_config, built_config=initial_built_config)
            net_change_gt = diff(gold_config=ground_truth_end_built_config, built_config=initial_built_config)

            generated_seq = output_obj["generated_seq"][0]
            # print(generated_seq)

            ground_truth_seq = output_obj["ground_truth_seq"][0].tolist() #[:-1]
            ground_truth_seq = prune_seq(ground_truth_seq, should_prune_seq(ground_truth_seq))
            # print(ground_truth_seq)

            def diff2actions(diff):
                placements = diff["gold_minus_built"]
                placements = list(map(
                    lambda block: add_action_type(block, "placement"),
                    placements
                ))

                removals = diff["built_minus_gold"]
                removals = list(map(
                    lambda block: add_action_type(block, "removal"), # TODO: color None?
                    removals
                ))

                return placements + removals

            # print("HERE")
            # print(generated_seq)
            # print(net_change_generated)
            # print("\n")
            # print(ground_truth_seq)
            # print(net_change_gt)
            # print("\n\n")

            net_change_generated = diff2actions(net_change_generated)
            # print(net_change_generated)
            net_change_gt = diff2actions(net_change_gt)
            # print(net_change_gt)
            # if len(net_change_generated):
            #     print(net_change_generated[0].__dict__)
            #     sys.exit(0)

            def get_pos_neg_count(generated_actions, gt_actions):
                net_change_generated = set(map(dict_to_tuple, generated_actions))
                net_change_gt = set(map(dict_to_tuple, gt_actions))

                fn = len(net_change_gt - net_change_generated)
                fp = len(net_change_generated - net_change_gt)
                tp = len(net_change_generated & net_change_gt)

                return fn, fp, tp

            a, b, c = get_pos_neg_count(net_change_generated, net_change_gt)
            fn += a
            fp += b
            tp += c

        # print("DONE!")
        # sys.exit(0)

        if args.compute_extra_stats:
            mean_edit_distance = np.mean(all_edit_distances)
            std_edit_distance = np.std(all_edit_distances)

            print('Mean edit distance of the network on the test set:', mean_edit_distance)
            print('SD of edit distances of the network on the test set:', std_edit_distance)

            mean_mean_edit_distance_per_alignment = np.mean(all_mean_edit_distances_per_alignment)
            std_mean_edit_distance_per_alignment = np.std(all_mean_edit_distances_per_alignment)

            print('Mean mean edit distance per alignment of the network on the test set:', mean_mean_edit_distance_per_alignment)
            print('SD of mean edit distances per alignment of the network on the test set:', std_mean_edit_distance_per_alignment)

        def compute_action_prf(fn, fp, tp):
            action_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            action_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            action_f1 = ((2 * action_precision * action_recall) / (action_precision + action_recall)) if (action_precision + action_recall) > 0 else 0.0

            return action_precision, action_recall, action_f1

        action_precision, action_recall, action_f1 = compute_action_prf(fn, fp, tp)

        print('Action precision of the network on the test set:', action_precision)
        print('Action recall of the network on the test set:', action_recall)
        print('Action F1 of the network on the test set:', action_f1)

        total = 0
        if args.compute_extra_stats:
            correct = 0
            correct_by_set = 0

            empty_seqs = []
            actions_only_seqs = []

            empty_seqs_gt = []
            actions_only_seqs_gt = []

        generated_seqs_lens = []
        ground_truth_seqs_lens = []

        for output_obj in generated_seqs:
            generated_seq = output_obj["generated_seq"][0]

            ground_truth_seq = output_obj["ground_truth_seq"][0].tolist() #[:-1]
            ground_truth_seq = prune_seq(ground_truth_seq, should_prune_seq(ground_truth_seq))

            total += 1

            if args.compute_extra_stats:
                # compare both for accuracy
                if generated_seq == ground_truth_seq:
                    correct += 1
                if set(generated_seq) == set(ground_truth_seq):
                    correct_by_set += 1

                # split into buckets -- action+, empty
                if generated_seq == []:
                    empty_seqs.append(output_obj)
                else:
                    actions_only_seqs.append(output_obj)

                # split into buckets -- action+, empty
                if ground_truth_seq == []:
                    empty_seqs_gt.append(output_obj)
                else:
                    actions_only_seqs_gt.append(output_obj)

            # track lengths
            generated_seqs_lens.append(len(generated_seq))
            ground_truth_seqs_lens.append(len(ground_truth_seq))

        if args.compute_extra_stats:
            accuracy = correct / total
            set_accuracy = correct_by_set / total
            print('Accuracy of the network on the test set:', accuracy)
            print('Set accuracy of the network on the test set:', set_accuracy)

            empty_seqs_fraction = len(empty_seqs)/len(generated_seqs)
            actions_only_seqs_fraction = len(actions_only_seqs)/len(generated_seqs)

            print('Fraction of empty_seqs in generated seqs:', empty_seqs_fraction)
            print('Fraction of actions_only_seqs in generated seqs:', actions_only_seqs_fraction)

            empty_seqs_gt_fraction = len(empty_seqs_gt)/len(generated_seqs)
            actions_only_seqs_gt_fraction = len(actions_only_seqs_gt)/len(generated_seqs)

            print('Fraction of empty_seqs_gt in generated seqs:', empty_seqs_gt_fraction)
            print('Fraction of actions_only_seqs_gt in generated seqs:', actions_only_seqs_gt_fraction)

        mean_generated_seq_length = np.mean(generated_seqs_lens)
        std_generated_seq_length = np.std(generated_seqs_lens)

        mean_ground_truth_seq_length = np.mean(ground_truth_seqs_lens)
        std_ground_truth_seq_length = np.std(ground_truth_seqs_lens)

        print('Mean generated seq length of the network on the test set:', mean_generated_seq_length)
        print('SD of generated seq length of the network on the test set:', std_generated_seq_length)

        print('Mean seq length of val data:', mean_ground_truth_seq_length)
        print('SD of seq length of val data:', std_ground_truth_seq_length)

        infeasible_seqs = 0
        infeasible_actions_fractions = []
        for output_obj in generated_seqs:
            action_feasibilities = output_obj["action_feasibilities"][0]
            # print(action_feasibilities)
            if False in action_feasibilities:
                infeasible_seqs += 1

            if len(action_feasibilities) > 0:
                infeasible_actions_fraction = len(list(filter(lambda x: x == False, action_feasibilities))) / len(action_feasibilities)
            else:
                infeasible_actions_fraction = 0.0
            infeasible_actions_fractions.append(infeasible_actions_fraction)

        infeasible_seqs_fraction = infeasible_seqs / total
        print('Infeasible seqs fraction of the network on the test set:', infeasible_seqs_fraction)

        infeasible_actions_fraction_macro_avg = sum(infeasible_actions_fractions) / len(infeasible_actions_fractions)
        print('Infeasible actions fraction (macro avg) of the network on the test set:', infeasible_actions_fraction_macro_avg)

        generated_seqs_formatted = json.dumps(list(map(format, generated_seqs)), indent=4, separators= (",\n", ": "), sort_keys=True)

        print("\n"+timestamp(), "Writing generated sentences to", print_dir(output_path, 6), "...\n")

        with open(output_path, 'w') as f:
            f.write('action_precision ' + str(action_precision) + "\n")
            f.write('action_recall ' + str(action_recall) + "\n")
            f.write('action_f1 ' + str(action_f1) + "\n")
            if args.compute_extra_stats:
                f.write('mean_edit_distance ' + str(mean_edit_distance) + "\n")
                f.write('std_edit_distance ' + str(std_edit_distance) + "\n")
                f.write('mean_mean_edit_distance_per_alignment ' + str(mean_mean_edit_distance_per_alignment) + "\n")
                f.write('std_mean_edit_distance_per_alignment ' + str(std_mean_edit_distance_per_alignment) + "\n")
                f.write('empty_seqs_fraction ' + str(empty_seqs_fraction) + "\n")
                f.write('actions_only_seqs_fraction ' + str(actions_only_seqs_fraction) + "\n")
            f.write('mean_generated_seq_length ' + str(mean_generated_seq_length) + "\n")
            f.write('std_generated_seq_length ' + str(std_generated_seq_length) + "\n")
            if args.compute_extra_stats:
                f.write('accuracy ' + str(accuracy) + "\n")
                f.write('set_accuracy ' + str(set_accuracy) +  "\n")
            f.write('infeasible_seqs_fraction ' + str(infeasible_seqs_fraction) + "\n")
            f.write('infeasible_actions_fraction_macro_avg ' + str(infeasible_actions_fraction_macro_avg) + "\n\n")
            if args.compute_extra_stats:
                f.write('empty_seqs_gt_fraction ' + str(empty_seqs_gt_fraction) + "\n")
                f.write('actions_only_seqs_gt_fraction ' + str(actions_only_seqs_gt_fraction) + "\n")
            f.write('mean_ground_truth_seq_length ' + str(mean_ground_truth_seq_length) + "\n")
            f.write('std_ground_truth_seq_length ' + str(std_ground_truth_seq_length) + "\n\n")
            f.write(generated_seqs_formatted)

        if to_print != None:
            with open(os.path.join(args.model_dir, 'raw_sentences'+args_sfx+'.txt'), 'w') as f:
                for line in to_print:
                    f.write(str(line)+'\n')

        return action_precision, action_recall, action_f1

        # def line_prepender(filename, line):
        #     with open(filename, 'r+') as f:
        #         content = f.read()
        #         f.seek(0, 0)
        #         f.write(line.rstrip('\r\n') + '\n' + content)
        #
        # line_prepender(output_path, 'action_f1 ' + str(action_f1))
        # line_prepender(output_path, 'action_recall ' + str(action_recall))
        # line_prepender(output_path, 'action_precision ' + str(action_precision))

    except KeyboardInterrupt:
        print("Quitting generation early.")
        with open(os.path.join(args.model_dir, 'fail.txt'), 'w') as f:
            f.write('Generation stopped early.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help='path for saved model to generate from')
    parser.add_argument('--saved_dataset_dir', type=str, default="../../data/saved_cwc_datasets/lower-builder_actions_only-no_diff-no_perspective_coords", help='path for saved dataset to use')
    parser.add_argument('--output_path', type=str, default=None, help='path for output file of generated sentences')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size for beam search decoding')
    parser.add_argument('--max_decoding_length', type=int, default=10, help='max iterations for decoder when decoding w/o ground truth inputs') # FIXME: Do not use hard coded string
    parser.add_argument('--masked_decoding', default=False, action='store_true', help='whether or not to use masked decoding to mask out infeasible actions')
    parser.add_argument("--development_mode", default=False, action="store_true", help="Whether or not to run in development mode, i.e., with less data")
    parser.add_argument('--regenerate_sentences', default=False, action='store_true', help='generate sentences for a model even if a generated sentences file already exists in its directory')
    parser.add_argument('--model_iteration', default='best_by_loss', help='iteration of model to be evaluated: "best" or "final"')
    parser.add_argument('--split', default='val', help='data split from which sentences should be generated')
    parser.add_argument('--disable_shuffle', default=False, action='store_true', help='disable shuffling of the data to be generated from')
    parser.add_argument("--eval_only", default=False, action="store_true", help="Whether or not to run eval only -- assuming that generated output has been pickled already -- also regenerate_sentences needs to be set typically")
    parser.add_argument("--compute_extra_stats", default=False, action="store_true", help="Whether or not to compute extra stats/metrics about generated seqs")

    parser.add_argument("--init_rngs", default=False, action="store_true", help="Whether or not to initialize RNGs at the start")

    args = parser.parse_args()

    print(timestamp(), args, "\n")

    # import cProfile, pstats, io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    # pr.enable()

    main(args)

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s)
    # ps.sort_stats(SortKey.CUMULATIVE)
    # ps.print_stats()
    # print(s.getvalue())
