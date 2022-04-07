import argparse, pprint as pp, torch, sys, numpy as np
from collections import defaultdict
import copy
sys.path.append('..')

from utils import *
from vocab import Vocabulary
from data_loader import CwCDataset, BuilderActionExample, Region

class BuilderActionsDataset(CwCDataset):
    def __init__(self, itemize_args, **kwargs):
        if not kwargs["items_only"]:
            # samples-only and jsons-only mode
            extra_args = ["items_only", "load_items", "dump_items", "development_mode", "split_questions"]
            kwargs_super = {i: kwargs[i] for i in kwargs if i not in extra_args}
            super(BuilderActionsDataset, self).__init__(**kwargs_super)
        else:
            # items-only/item batches-only mode
            if kwargs["load_items"]:
                # FIXME: Do we need to match conf against itemize_args? Or get the following from the conf?

                # set itemize_args
                self.num_prev_utterances = itemize_args.num_prev_utterances
                self.use_builder_actions = itemize_args.use_builder_actions
                self.add_perspective_coords = itemize_args.add_perspective_coords
                self.num_prev_utterances_by_heuristic = itemize_args.num_prev_utterances_by_heuristic
                self.num_prev_utterances_until_last_architect = itemize_args.num_prev_utterances_until_last_architect
                self.num_prev_utterances_until_last_action = itemize_args.num_prev_utterances_until_last_action
                self.include_empty_channel = itemize_args.include_empty_channel
                self.neighborhood_window_size = itemize_args.neighborhood_window_size
                self.add_action_history_weight = itemize_args.add_action_history_weight
                self.action_history_weighting_scheme = itemize_args.action_history_weighting_scheme
                self.concatenate_action_history_weight = itemize_args.concatenate_action_history_weight
                self.two_dim_posterior = itemize_args.two_dim_posterior

                if kwargs["split"] == "train": # items
                    print("Loading items...")

                    with open(os.path.join(kwargs["saved_dataset_dir"], "train-items.pkl"), 'rb') as f:
                        self.items = torch.load(f, map_location="cpu") # NOTE: always load onto cpu

                    if kwargs["development_mode"]:
                        self.items = self.items[:10]
                else: # item batches
                    print("Loading item batches...")

                    with open(os.path.join(kwargs["saved_dataset_dir"], kwargs["split"] + "-item_batches.pkl"), 'rb') as f:
                        self.item_batches = torch.load(f, map_location="cpu") # NOTE: always load onto cpu

                    if kwargs["development_mode"]:
                        self.item_batches = self.item_batches[:5]
            else:
                # load samples
                extra_args = ["items_only", "load_items", "dump_items", "development_mode", "split_questions"]
                kwargs_super = {i: kwargs[i] for i in kwargs if i not in extra_args}
                super(BuilderActionsDataset, self).__init__(**kwargs_super)

                # set itemize_args
                self.num_prev_utterances = itemize_args.num_prev_utterances
                self.use_builder_actions = itemize_args.use_builder_actions
                self.add_perspective_coords = itemize_args.add_perspective_coords
                self.num_prev_utterances_by_heuristic = itemize_args.num_prev_utterances_by_heuristic
                self.num_prev_utterances_until_last_architect = itemize_args.num_prev_utterances_until_last_architect
                self.num_prev_utterances_until_last_action = itemize_args.num_prev_utterances_until_last_action
                self.include_empty_channel = itemize_args.include_empty_channel
                self.neighborhood_window_size = itemize_args.neighborhood_window_size
                self.add_action_history_weight = itemize_args.add_action_history_weight
                self.action_history_weighting_scheme = itemize_args.action_history_weighting_scheme
                self.concatenate_action_history_weight = itemize_args.concatenate_action_history_weight
                self.two_dim_posterior = itemize_args.two_dim_posterior
                self.split_questions = kwargs['split_questions']

                # generate items
                if kwargs["split"] == "train": # items
                    print("Generating items...")

                    self.items = []
                    for i in range(len(self)):
                        if kwargs["development_mode"] and i == 10:
                            break
                        self.items.append(self.__getitem__(i))

                    # dump if needed
                    if kwargs["dump_items"]:
                        torch.save(self.items, os.path.join(kwargs["saved_dataset_dir"], "train-items.pkl")) # NOTE: generated on cpu, no need to move to cpu
                else: # item batches
                    print("Generating item batches...")

                    if kwargs['split_questions']:
                        print("Splitting on question data")
                        self.samples = self.split_qs()

                    loader = self.get_data_loader(
                        batch_size=itemize_args.batch_size, shuffle=False, num_workers=itemize_args.num_workers
                    )

                    self.item_batches = []
                    for i, data in enumerate(loader, 0):
                        if kwargs["development_mode"] and i == 5:
                            break
                        self.item_batches.append(data)

                    # dump if needed
                    if kwargs["dump_items"]:
                        torch.save(self.item_batches, os.path.join(kwargs["saved_dataset_dir"], kwargs["split"] + "-item_batches.pkl")) # NOTE: generated on cpu, no need to move to cpu

                # dump additional info
                if kwargs["dump_items"]:
                    print("\nSaving itemize_args config file ...\n")
                    with open(os.path.join(kwargs["saved_dataset_dir"], kwargs["split"] + "-itemize_args-config.txt"), "w") as f:
                        args_dict = vars(itemize_args)
                        for param in args_dict:
                            f.write(param.ljust(20)+"\t"+str(args_dict[param])+"\n")

                    print("\nSaving git commit hashes ...\n")
                    write_commit_hashes("../..", kwargs["saved_dataset_dir"], filepath_modifier="_items_" + kwargs["split"])

    def is_sorted(self, lst):
        """
        Check if the provided list is sorted numberically (recursively)
        """
        if len(lst) == 1:
            return True
        return lst[0] <= lst[1] and self.is_sorted(lst[1:])

    def create_question_sample(self, sample, utterances):
        """
        Create a new sample object based on another, but with a question label
        """
        new_sample = copy.copy(sample)
        new_sample['prev_utterances'] = copy.copy(utterances)
        new_sample['question'] = True
        # If we remove the next builder actions we destroy the __getitem__
        # interface.. However, we should probably do it to make sure we don't
        # provide the agent with too much information?
        # new_sample['next_builder_actions'] = []

        return new_sample

    def check_zeroth_sample(self, sample):
        """
        Check if the first step in an episode contains questions from the
        builder and create additional samples if there are.
        """
        intermediate_samples = []
        recreate_utterances = []
        for utterance in sample['prev_utterances']:
            if utterance['speaker'] == 'Builder' and '?' in utterance['utterance']:
                new_sample = self.create_question_sample(sample, recreate_utterances)
                intermediate_samples.append(new_sample)
            recreate_utterances.append(utterance)
        sample['question'] = False
        intermediate_samples.append(sample)
        return intermediate_samples

    def check_step_diff(self, sample_prev, sample_new):
        """
        Check if there were questions asked in the chat difference between two
        steps (_prev and _new), and create additional samples if there are.
        """
        chat_prev_len = len(sample_prev['prev_utterances'])
        chat_diff = sample_new['prev_utterances'][chat_prev_len:]
        intermediate_samples = []
        recreate_utterances = sample_prev['prev_utterances']
        for utterance in chat_diff:
            if utterance['speaker'] == 'Builder' and '?' in utterance['utterance']:
                new_sample = self.create_question_sample(sample_new, recreate_utterances)
                intermediate_samples.append(new_sample)
            recreate_utterances.append(utterance)
        sample_new['question'] = False
        intermediate_samples.append(sample_new)
        return intermediate_samples

    def extract_questions_in_episode(self, samples_in_episode):
        """
        Extract samples from episode steps where the builder asked a question
        """
        sample_ids = [x['sample_id'] for x in samples_in_episode]
        assert self.is_sorted(sample_ids) # episode should be ordered

        new_samples = []
        zeroth_samples = self.check_zeroth_sample(samples_in_episode[0])
        new_samples.extend(zeroth_samples)
        for i in range(len(samples_in_episode)-1):
            intermediate_samples = self.check_step_diff(samples_in_episode[i], samples_in_episode[i+1])
            new_samples.extend(intermediate_samples)
        return new_samples

    def split_qs(self):
        """
        Split the dataset on questions in the chat asked by the builder agent

        Creates new (empty action) samples for question actions, inflating the
        dataset.
        """
        print(f"Length before splitting: {len(self.samples)}")
        episode_data = defaultdict(lambda: [])
        for sample in self.samples:
            episode_data[sample['json_id']].append(sample)
        new_samples = []
        for _, episode in episode_data.items():
            new_episode_samples = self.extract_questions_in_episode(episode)
            new_samples.extend(new_episode_samples)
        print(f"Length after splitting: {len(new_samples)}")
        return new_samples

    def __getitem__(self, idx):
        """ Computes the tensor representations of a sample """
        orig_sample = self.samples[idx]

        all_actions = orig_sample["next_builder_actions"]
        perspective_coords = orig_sample["perspective_coordinates"]

        initial_prev_config_raw = all_actions[0].prev_config
        initial_action_history_raw = all_actions[0].action_history
        end_built_config_raw = all_actions[-1].built_config

        all_actions_reprs = list(map(lambda x: self.get_repr(x, perspective_coords), all_actions))
        all_grid_repr_inputs = list(map(lambda x: x[0], all_actions_reprs))
        all_outputs = list(map(lambda x: x[1], all_actions_reprs))

        all_actions_repr_inputs = list(map(f2, all_actions))

        start_action_repr = torch.Tensor([0] * 11)

        stop_action = BuilderActionExample(
            action = None,
            built_config = all_actions[-1].built_config,
            prev_config = all_actions[-1].built_config,
            action_history = all_actions[-1].action_history + [all_actions[-1].action]
        )
        stop_action_repr = self.get_repr(stop_action, perspective_coords)
        stop_action_grid_repr_input = stop_action_repr[0]
        stop_action_output_label = stop_action_repr[1]

        dec_inputs_1 = all_grid_repr_inputs + [stop_action_grid_repr_input]
        dec_inputs_2 = [start_action_repr] + all_actions_repr_inputs

        dec_outputs = all_outputs + [stop_action_output_label]

        # Encoder inputs
        utterances_to_add = []

        if self.num_prev_utterances_by_heuristic:
            found = 0
            for index in range(len(orig_sample["prev_utterances"])-1, -1, -1):
                # iterate in reverse
                item = orig_sample["prev_utterances"][index]
                speaker = item["speaker"]
                utterance = item["utterance"]

                if index == len(orig_sample["prev_utterances"])-1:
                    next_item = None
                else:
                    next_item = orig_sample["prev_utterances"][index+1]

                if (next_item != None) and ("<builder_" in utterance[0] and not "<builder_" in next_item["utterance"][0]):
                    found += 1
                    if found == 2:
                        break

                if "mission has started ." in " ".join(utterance) and 'Builder' in speaker:
                    continue

                if "<builder_" in utterance[0]:
                    if self.use_builder_actions:
                        utterances_to_add.insert(0, item)
                else:
                    utterances_to_add.insert(0, item)
        elif self.num_prev_utterances_until_last_architect:
            for index in range(len(orig_sample["prev_utterances"])-1, -1, -1):
                # iterate in reverse
                item = orig_sample["prev_utterances"][index]
                speaker = item["speaker"]
                utterance = item["utterance"]

                if speaker == "Architect":
                    utterances_to_add.insert(0, item)
                    break

                if "mission has started ." in " ".join(utterance) and 'Builder' in speaker:
                    continue

                if "<builder_" in utterance[0]:
                    if self.use_builder_actions:
                        utterances_to_add.insert(0, item)
                else:
                    utterances_to_add.insert(0, item)
        elif self.num_prev_utterances_until_last_action:
            found = 0
            for index in range(len(orig_sample["prev_utterances"])-1, -1, -1):
                # iterate in reverse
                item = orig_sample["prev_utterances"][index]
                speaker = item["speaker"]
                utterance = item["utterance"]

                if index == len(orig_sample["prev_utterances"])-1:
                    next_item = None
                else:
                    next_item = orig_sample["prev_utterances"][index+1]

                if (next_item != None) and ("<builder_" in utterance[0] and not "<builder_" in next_item["utterance"][0]):
                    found += 1
                    if found == 1:
                        break

                if "mission has started ." in " ".join(utterance) and 'Builder' in speaker:
                    continue

                if "<builder_" in utterance[0]: # should never fire
                    if self.use_builder_actions:
                        utterances_to_add.insert(0, item)
                else:
                    utterances_to_add.insert(0, item)
        else:
            i = 0
            utterances_idx = len(orig_sample["prev_utterances"])-1
            while i < self.num_prev_utterances:
                if utterances_idx < 0:
                    break

                prev = orig_sample["prev_utterances"][utterances_idx]
                speaker = prev["speaker"]
                utterance = prev["utterance"]

                if "<builder_" in utterance[0]:
                    if self.use_builder_actions:
                        utterances_to_add.insert(0, prev)
                    i -= 1

                elif "mission has started ." in " ".join(utterance) and 'Builder' in speaker:
                    i -= 1

                else:
                    utterances_to_add.insert(0, prev)

                utterances_idx -= 1
                i += 1

        prev_utterances = []

        for prev in utterances_to_add:
            speaker = prev["speaker"]
            utterance = prev["utterance"]

            if "<dialogue>" in utterance[0]:
                prev_utterances.append(self.encoder_vocab('<dialogue>'))

            elif "<builder_" in utterance[0]:
                if self.use_builder_actions:
                    prev_utterances.append(self.encoder_vocab(utterance[0]))

            else:
                start_token = self.encoder_vocab('<architect>') if 'Architect' in speaker else self.encoder_vocab('<builder>')
                end_token = self.encoder_vocab('</architect>') if 'Architect' in speaker else self.encoder_vocab('</builder>')
                prev_utterances.append(start_token)
                prev_utterances.extend(self.encoder_vocab(token) for token in utterance)
                prev_utterances.append(end_token)
        if self.split_questions:
            return (
                torch.Tensor(prev_utterances),
                torch.stack(dec_inputs_1),
                torch.stack(dec_inputs_2),
                torch.Tensor(dec_outputs),
                RawInputs(initial_prev_config_raw, initial_action_history_raw, end_built_config_raw, perspective_coords),
                orig_sample['question']
            )
        else:
            return (
                torch.Tensor(prev_utterances),
                torch.stack(dec_inputs_1),
                torch.stack(dec_inputs_2),
                torch.Tensor(dec_outputs),
                RawInputs(initial_prev_config_raw, initial_action_history_raw, end_built_config_raw, perspective_coords),
            )

    def collate_fn(self, data):
        # NOTE: assumes batch size = 1
        if self.split_questions:
            prev_utterances, dec_inputs_1, dec_inputs_2, dec_outputs, raw_inputs, question_label = zip(*data)
        else:
            prev_utterances, dec_inputs_1, dec_inputs_2, dec_outputs, raw_inputs = zip(*data)

        def merge_text(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # prev_utterances = torch.stack(prev_utterances)
        prev_utterances, prev_utterances_lengths = merge_text(prev_utterances)
        dec_inputs_1 = torch.stack(dec_inputs_1)
        dec_inputs_2 = torch.stack(dec_inputs_2)
        dec_outputs = torch.stack(dec_outputs)

        return (
            EncoderInputs(prev_utterances, prev_utterances_lengths),
            dec_inputs_1,
            dec_inputs_2,
            dec_outputs,
            raw_inputs[0]
        )

    def get_repr(self, builder_action, perspective_coords):
        # temporary fix: floats in configs # TODO: make sure this should be commented out
        # config = builder_action.built_config
        # for block in config:
        #     for key in ['x', 'y', 'z']:
        #         block[key] = int(block[key])

        config = builder_action.prev_config
        for block in config:
            for key in ['x', 'y', 'z']:
                block[key] = int(block[key])

        cell_samples = split_orig_sample(builder_action)

        # cell_samples_reprs = []
        cell_repr_size = 6
        if self.include_empty_channel:
            cell_repr_size += 1
        if self.add_action_history_weight and self.concatenate_action_history_weight:
            cell_repr_size += 1

        cell_samples_reprs_4d = torch.zeros([cell_repr_size, 11, 9, 11]) # [repr, x, y, z]
        cell_samples_labels = []

        for sample in cell_samples:
            cell = Region(
                x_min = sample["x"], x_max = sample["x"], y_min = sample["y"], y_max = sample["y"], z_min = sample["z"], z_max = sample["z"]
            )

            def get_current_state(cell, built_config):
                current_state = "empty"

                for block in built_config:
                    if cell.x_min == block["x"] and cell.y_min == block["y"] and cell.z_min == block["z"]:
                        current_state = block["type"]
                        break

                return current_state
            current_state = get_current_state(cell, sample["built_config"])
            def get_action_history_weight(cell, action_history):
                if self.add_action_history_weight:
                    # get action history weight
                    action_history_weight = 0.0

                    for i in range(len(action_history) - 1, -1, -1):
                        # i -- 4, 3, 2, 1, 0
                        action = action_history[i]
                        if cell.x_min == action.block["x"] and cell.y_min == action.block["y"] and cell.z_min == action.block["z"]:
                            if self.action_history_weighting_scheme == "smooth":
                                action_history_weight = (i + 1)/len(action_history)
                            elif self.action_history_weighting_scheme == "step":
                                action_history_weight = i - (len(action_history) - 1) + 5
                                action_history_weight = np.maximum(action_history_weight, 0.0)
                            break
                else:
                    action_history_weight = None

                return action_history_weight
            action_history_weight = get_action_history_weight(cell, sample["builder_action_history"])

            # output label
            if sample["action_type"] == "placement":
                # print("placement")
                output_label = type2id[sample["block_type"]]
            elif sample["action_type"] == "removal":
                # print("removal")
                output_label = len(type2id)
            else: # None case -- no action
                # print("none")
                output_label = len(type2id) + 1

            # convert data to tensors
            def f(current_cell_state):
                if current_cell_state in type2id:
                    index = type2id[current_cell_state]
                else:
                    if self.include_empty_channel:
                        index = len(type2id) # 6
                    else:
                        index = None

                if self.include_empty_channel:
                    vec_length = len(type2id) + 1
                else:
                    vec_length = len(type2id)

                one_hot_vec = [0] * vec_length
                if index != None:
                    one_hot_vec[index] = 1

                return one_hot_vec

            current_cell_state_one_hot_vec = f(current_state)

            action_history_weight_vec = [action_history_weight]

            def get_joint_repr(vec1, vec2):
                if not vec2 == [None]:
                    # get joint repr for a region
                    if self.concatenate_action_history_weight:
                        # concatenate
                        new_vec = vec1 + vec2
                    else:
                        # find the 1.0 in vec1 if any and replace it by vec2[0] if latter is >= 1.0
                        new_vec = list(map(
                            lambda x: x if x == 0 or (x == 1 and vec2[0] == 0) else vec2[0],
                            vec1
                        ))

                    return new_vec
                else:
                    return vec1
            repr_vec = get_joint_repr(current_cell_state_one_hot_vec, action_history_weight_vec)

            # cell_samples_reprs.append(repr_vec)
            cell_samples_reprs_4d[:, sample["x"]+5, sample["y"]-1, sample["z"]+5] = torch.Tensor(repr_vec) # offsets needed to map to range [0, something]
            cell_samples_labels.append(output_label)

        action_cell_index = None
        action_cell_label = None
        for i, label in enumerate(cell_samples_labels):
            if label < 7:
                action_cell_index = i
                action_cell_label = label
                break

        if action_cell_label != None:
            output_label = action_cell_index * 7 + action_cell_label # act
        else:
            # stop
            assert builder_action.is_stop_token()
            output_label = stop_action_label

        # grid_repr = [y for x in cell_samples_reprs for y in x]

        if self.add_perspective_coords:
            cell_samples_reprs_4d = torch.cat([cell_samples_reprs_4d, perspective_coords/10], dim=0)

        return (
            cell_samples_reprs_4d,
            output_label
        )

# UTILS
def split_orig_sample(action):
    builder_action = action.action # None if stop action
    new_samples = []

    # generate all possible grid locations
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            for z in range(z_min, z_max + 1):
                new_sample = {
                    "x": x,
                    "y": y,
                    "z": z
                }

                # what action happened at this loc
                if isinstance(builder_action, BuilderAction) and builder_action.block["x"] == x and builder_action.block["y"] == y and builder_action.block["z"] == z:
                    # fails if builder_action is None, i.e., stop token
                    # hit
                    new_sample["action_type"] = builder_action.action_type # placement or removal
                    new_sample["block_type"] = builder_action.block["type"]
                else:
                    # miss -- no action
                    new_sample["action_type"] = None
                    new_sample["block_type"] = None

                new_samples.append(new_sample)

    # copy over other stuff
    for new_sample in new_samples:
        # for key in ["prev_utterances", "built_config", "last_action", "builder_position", "perspective_coordinates", "from_aug_data", "json_id", "sample_id", "builder_action_history"]:
        #     new_sample[key] = orig_sample[key]
        new_sample["built_config"] = action.prev_config
        new_sample["builder_action_history"] = action.action_history

    return new_samples

class EncoderInputs:
    def __init__(self, prev_utterances, prev_utterances_lengths):
        self.prev_utterances = prev_utterances # previous utterances
        self.prev_utterances_lengths = prev_utterances_lengths

class RawInputs:
    def __init__(self, initial_prev_config_raw, initial_action_history_raw, end_built_config_raw, perspective_coords):
        self.initial_prev_config_raw = initial_prev_config_raw
        self.initial_action_history_raw = initial_action_history_raw
        self.end_built_config_raw = end_built_config_raw
        self.perspective_coords = perspective_coords

if __name__ == '__main__':
    """
        Script to generate items/item batches for the BAP task
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', default='train', help='dataset split')

    parser.add_argument('--load_items', default=False, action='store_true', help='load items/item batches')
    parser.add_argument('--dump_items', default=False, action='store_true', help='generate and dump items/item batches')

    parser.add_argument('--saved_dataset_dir', type=str, default="../../data/saved_cwc_datasets/lower-builder_actions_only-no_diff-no_perspective_coords", help='path for saved dataset to use')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument("--development_mode", default=False, action="store_true", help="Whether or not to run in development mode, i.e., with less data")

    parser.add_argument("--include_empty_channel", default=True, action="store_true", help="Whether to add an empty channel in input repr")
    parser.add_argument('--neighborhood_window_size', type=int, default=1, help='size of window to conside for representing neighborhood of a cell')
    parser.add_argument("--add_action_history_weight", default=True, action="store_true", help="Whether to add an extra bit for action history weights in input repr")
    parser.add_argument('--action_history_weighting_scheme', type=str, default="step", help='type of action weighting scheme to use')
    parser.add_argument("--concatenate_action_history_weight", default=True, action="store_true", help="Whether to concatenate or incorporate into the vector representing the current block in a neighborhood cell")

    parser.add_argument("--two_dim_posterior", default=True, action="store_true", help="Whether to only compute 2 dim placement/removal posterior")

    parser.add_argument('--encoder_vocab_path', type=str, default='../../vocabulary/glove.42B.300d-lower-1r-speaker-builder_actions-oov_as_unk-all_splits/vocab.pkl', help='path for encoder vocabulary wrapper')

    parser.add_argument('--num_prev_utterances', type=int, default=3, help='number of previous utterances to use as input')
    parser.add_argument('--num_prev_utterances_by_heuristic', default=False, action='store_true', help='whether to decide number of previous utterances to use as input based on the heuristic')
    parser.add_argument('--num_prev_utterances_until_last_architect', default=False, action='store_true', help='whether to include previous utterances until last Architect utterance')
    parser.add_argument('--num_prev_utterances_until_last_action', default=False, action='store_true', help='whether to include previous utterances until the last Builder action')
    parser.add_argument('--use_builder_actions', default=False, action='store_true', help='include builder action tokens in the dialogue history')

    parser.add_argument('--add_perspective_coords', default=False, action='store_true', help='whether or not to include perspective coords in world state repr')
    parser.add_argument('--split_questions', default=False, action='store_true', help='split the dataset also on question actions')

    args = parser.parse_args()

    initialize_rngs(args.seed, torch.cuda.is_available())

    if args.use_builder_actions and 'builder_actions' not in args.encoder_vocab_path:
        print("Error: you specified to use builder action tokens in the dialogue history, but they do not exist in the encoder's vocabulary.")
        sys.exit(0)

    if not args.use_builder_actions and 'builder_actions' in args.encoder_vocab_path:
        print("Warning: you specified not to use builder action tokens, but your encoder vocabulary contained them; resetting vocabulary to default: ../../vocabulary/glove.42B.300d-lower-1r-speaker-oov_as_unk-all_splits/vocab.pkl")
        args.encoder_vocab_path = '../../vocabulary/glove.42B.300d-lower-1r-speaker-oov_as_unk-all_splits/vocab.pkl'

    # load the vocabularies
    with open(args.encoder_vocab_path, 'rb') as f:
        print(timestamp(), "Loading encoder vocabulary from", print_dir(args.encoder_vocab_path, 3), "...")
        encoder_vocab = pickle.load(f)
        print(timestamp(), "Successfully loaded encoder vocabulary.\n")

    dataset = BuilderActionsDataset(
        itemize_args=args,

        split=args.split,
        saved_dataset_dir=args.saved_dataset_dir,
        load_dataset=True,
        encoder_vocab=encoder_vocab,

        items_only=True,
        load_items=args.load_items,
        dump_items=args.dump_items,
        development_mode=args.development_mode,
        split_questions=args.split_questions
    )

    # Format of item in dataset
    #     torch.Tensor(prev_utterances),
    #     torch.stack(dec_inputs_1),
    #     torch.stack(dec_inputs_2),
    #     torch.Tensor(dec_outputs),
    #     RawInputs(initial_prev_config_raw, initial_action_history_raw, end_built_config_raw, perspective_coords)
    #     <question label>
    print(dataset[0][0])
    print(dataset[0][-1])
    print(dataset[0][-2].initial_prev_config_raw)
    print(dataset[0][-2].end_built_config_raw)