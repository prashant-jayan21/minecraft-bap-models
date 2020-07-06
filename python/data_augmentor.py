import os, re, json, itertools, random, sys, time, argparse, numpy as np, torch
from os.path import join, isdir
from collections import Counter
import xml.etree.ElementTree as ET

from utils import type2id, color_regex, tokenize, timestamp, initialize_rngs, write_commit_hashes
from diff import is_feasible_config, generate_perturbation, build_region_specs, PerturbedConfig, perturb_builder_position

# MAIN CLASS

class DataAugmentor:
	"""
		Class responsible for dialog level augmentation
	"""
	def __init__(self, data_path, gold_configs_dir, augmentation_factor, out_data_path, substitutions_lexicon_file, lower, skip_paraphrasing, skip_color_permutation, skip_spatial_transform):
		self.data_path = data_path
		self.gold_configs_dir = gold_configs_dir
		self.augmentation_factor = augmentation_factor
		self.out_data_path = out_data_path
		self.lower = lower
		self.skip_paraphrasing = skip_paraphrasing
		self.skip_color_permutation = skip_color_permutation
		self.skip_spatial_transform = skip_spatial_transform
		# parse
		with open(substitutions_lexicon_file) as f:
			self.substitutions_lexicon = json.loads(f.read())

	def generate(self):
		"""
			Process dialogs one-by-one and mimic directory structure of data/logs when writing synthetic dialogs to file
		"""
		all_data_root_dirs = filter(lambda x: isdir(join(self.data_path, x)), os.listdir(self.data_path))

		augmentations_per_gold_config = Counter()

		for data_root_dir in all_data_root_dirs:
			print("\n")
			print(data_root_dir)
			logs_root_dir = join(self.data_path, data_root_dir, "logs")

			all_log_dirs = filter(lambda x: isdir(join(logs_root_dir, x)), os.listdir(logs_root_dir))

			for log_dir in all_log_dirs:
				# print("\t", log_dir)

				config_name = re.sub(r"B\d+-A\d+-|-\d\d\d\d\d\d\d+", "", log_dir)
				config_xml_file = join(self.gold_configs_dir, config_name + ".xml")
				config_structure = get_gold_config(config_xml_file)

				logfile = join(logs_root_dir, log_dir, "postprocessed-observations.json")
				with open(logfile) as f:
					observations_json = json.loads(f.read())

				# generate synthetic dialogs and gold configs
				synthetic_observations_objs, synthetic_gold_configs = self.augment(
					observations_json, log_dir, config_structure, config_name, augmentations_per_gold_config
				)

				# write both to file
				for synthetic_observations_obj in synthetic_observations_objs:
					json_filename = join(self.out_data_path, "logs", data_root_dir, "logs", synthetic_observations_obj.log_dir, "postprocessed-observations.json")
					os.makedirs(os.path.dirname(json_filename), exist_ok=True)
					with open(json_filename, "w") as f:
						json.dump(synthetic_observations_obj.json_content, f)

				for synthetic_gold_config in synthetic_gold_configs:
					xml_filename = join(self.out_data_path, "gold-configurations", synthetic_gold_config.filename)
					os.makedirs(os.path.dirname(xml_filename), exist_ok=True)
					with open(xml_filename, "w") as f:
						string_to_write = "\n".join(list(map(lambda x: ET.tostring(x, "unicode"), synthetic_gold_config.xml_content)))
						f.write(string_to_write)

		print("\nSaving git commit hashes ...\n")
		write_commit_hashes("..", self.out_data_path)

	def augment(self, observations_json, log_dir, config_structure, config_name, augmentations_per_gold_config):
		"""
			Args:
				observations_json: All observations in a dialog
				config_structure: The gold config as a list of blocks -- each block is a dict

			Returns:
				A list of synthetic dialogs (each is again in the same format as observations_json)
				A list of synthetic gold configs (each is again in the same format as config_structure)
		"""
		synthetic_observations_objs = []
		synthetic_gold_configs = []

		for i in range(self.augmentation_factor):
			# print("Augmentation", i)

			if not self.skip_color_permutation:
				# substituting colors
				random_permutation = random.choice(all_color_permutations)

				new_observations_json, new_config_structure = apply_permutation(random_permutation, observations_json, config_structure, self.lower)
			else:
				random_permutation = tuple([k for k, v in sorted(type2id.items(), key=lambda x: x[1])])

				new_observations_json, new_config_structure = apply_permutation(random_permutation, observations_json, config_structure, self.lower)

			if not self.skip_paraphrasing:
				# paraphrasing utterances
				utterances = new_observations_json["WorldStates"][-1]["ChatHistory"]

				def get_random_new_utterance(utterance):
					all_tokens = utterance
					starting_system_token = all_tokens[0]
					all_tokens = all_tokens[1:]

					def get_subs(token):
						# map token to list of all possible substitutions
						token_substitutions = [token]
						if token in self.substitutions_lexicon:
							token_substitutions += self.substitutions_lexicon[token]
						return token_substitutions

					# map each token to a list of it's substitutions including itself
					substitutions_list = list(map(get_subs, all_tokens))

					# sample
					all_tokens_substituted = list(map(lambda x: random.choice(x), substitutions_list))

					all_tokens_substituted = [starting_system_token] + all_tokens_substituted

					return all_tokens_substituted

				new_utterances_tokenized = list(map(get_random_new_utterance, utterances))

				def paraphrase_entire_log(observations_json, new_chat_history_tokenized):
					for world_state in observations_json["WorldStates"]:
						chat_history = world_state["ChatHistory"]
						for i in range(len(chat_history)):
							chat_history[i] = new_chat_history_tokenized[i]
						world_state["ChatHistoryTokenized"] = world_state.pop("ChatHistory")

				paraphrase_entire_log(new_observations_json, new_utterances_tokenized)
			else:
				for world_state in new_observations_json["WorldStates"]:
					world_state["ChatHistoryTokenized"] = world_state.pop("ChatHistory")

			# spatial transformations of built config and builder position
			# transformation = random.choice(transformations)
			#
			# new_observations_json = apply_transformation(new_observations_json, transformation)

			if not self.skip_spatial_transform:
				# shuffle transformations
				transformations_shuffled = random.sample(transformations, k=len(transformations))

				# pick first feasible one
				def pick_and_apply():
					for transformation in transformations_shuffled:
						# check feasibility
						is_feasible, json_to_return = is_feasible_transformation(transformation, new_observations_json)

						if is_feasible:
							# import pprint
							# print(log_dir)
							# pprint.PrettyPrinter(indent=4).pprint((transformation.translation, transformation.rot_target))
							return json_to_return

					# return None
					print("No feasible transformation found!")
					print(log_dir)
					print(config_name)
					print("\n")

					return new_observations_json

				new_observations_json = pick_and_apply()
			else:
				pass

			# formatting results
			new_config_name = config_name + "_" + str(augmentations_per_gold_config[config_name])
			new_log_dir = log_dir.replace(config_name, new_config_name)

			augmentations_per_gold_config[config_name] += 1

			synthetic_observations_objs.append(SyntheticObservations(log_dir=new_log_dir, json_content=new_observations_json))
			synthetic_gold_configs.append(SyntheticGoldConfig(filename=new_config_name + ".xml", xml_content=new_config_structure))

		return synthetic_observations_objs, synthetic_gold_configs

# AUGMENTATION METHODS

all_colors = list(type2id.keys())
all_color_permutations = list(itertools.permutations(all_colors))

assert len(all_color_permutations) == 720

def generate_perturbations(no_translation=False):
	if no_translation:
		all_x_values = [0]
		all_z_values = [0]
	else:
		all_x_values = [i for i in range(build_region_specs["x_min_build"] - 5, build_region_specs["x_max_build"] + 1 + 5)]
		all_z_values = [i for i in range(build_region_specs["z_min_build"] - 5, build_region_specs["z_max_build"] + 1 + 5)]

	all_rot_values = [0, 90, 180, -90]

	perturbations = []

	for x in all_x_values:
		for z in all_z_values:
			for rot in all_rot_values:
				perturbation = PerturbedConfig(
					perturbed_config=None,
					x_target=x, z_target=z, rot_target=rot,
					rot_axis_pivot=(0, 0, 0), translation=(x, z),
					original_config=None
				)
				perturbations.append(perturbation)

	return perturbations

transformations = generate_perturbations(no_translation=True)

def is_feasible_transformation(transformation, observations_json):
	"""
		Args:
			transformation: PerturbedConfig

		Output:
			is_feasible: B_f & B_f-1 ... & B_0
	"""
	def reformat(block): # TODO: remove redundancy
		return {
			"x": block["AbsoluteCoordinates"]["X"],
			"y": block["AbsoluteCoordinates"]["Y"],
			"z": block["AbsoluteCoordinates"]["Z"],
			"type": block["Type"]
		}

	# get all B_i's
	all_built_configs = [] # first is oldest
	for world_state in observations_json["WorldStates"]:
		built_config = list(map(reformat, world_state["BlocksInGrid"]))
		all_built_configs.append(built_config)

	all_built_configs.reverse() # first is newest # NOTE: mutation

	# do AND op
	is_feasible = True
	for built_config in all_built_configs:
		new_built_config = generate_perturbation(
			config=built_config,
			x_target=transformation.x_target, z_target=transformation.z_target,
			rot_target=transformation.rot_target,
			gold_config=None
		).perturbed_config

		if is_feasible_config(new_built_config):
			continue
		else:
			is_feasible = False
			break

	# apply transformation if needed
	if is_feasible:
		new_observations_json = apply_transformation(observations_json, transformation)
	else:
		new_observations_json = None

	return is_feasible, new_observations_json # (True, a new json) OR (False, None)

def apply_permutation(permutation, observations_json, config_structure, lower):
	# define color mapping

	color_map = {}
	for key, value in type2id.items():
		color_map[key] = permutation[value]

	# apply color mapping to gold config

	def f(block):
		parsed_color = color_regex.findall(block.attrib["type"])[0]
		new_color = color_map[parsed_color]
		new_color_for_xml =  "cwcmod:cwc_" + new_color + "_rn"

		new_block_tag = block.tag
		new_block_attrib = block.attrib.copy()
		new_block_attrib["type"] = new_color_for_xml
		new_block = ET.Element(new_block_tag, new_block_attrib)

		return new_block

	new_config_structure = list(map(f, config_structure))

	# apply color mapping to dialog observations

	new_observations_json = {}

	new_observations_json["NumFixedViewers"] = observations_json["NumFixedViewers"]
	new_observations_json["TimeElapsed"] = observations_json["TimeElapsed"]

	def g(world_state):
		new_world_state = {}

		new_world_state["BuilderPosition"] = world_state["BuilderPosition"]
		new_world_state["Timestamp"] = world_state["Timestamp"]
		new_world_state["ScreenshotPath"] = world_state["ScreenshotPath"]

		def f(color):
			parsed_color_str = color_regex.findall(color["Type"])[0]
			new_color_str = color_map[parsed_color_str]
			new_color_str =  "cwcmod:cwc_" + new_color_str + "_rn"

			new_color = color.copy()
			new_color["Type"] = new_color_str

			return new_color

		if world_state["BuilderInventory"] is not None:
			new_world_state["BuilderInventory"] = list(map(f, world_state["BuilderInventory"]))
		else:
			new_world_state["BuilderInventory"] = None

		def h(block):
			new_block = {}

			new_block["PerspectiveCoordinates"] = block["PerspectiveCoordinates"]
			new_block["AbsoluteCoordinates"] = block["AbsoluteCoordinates"]

			parsed_color_str = color_regex.findall(block["Type"])[0]
			new_color_str = color_map[parsed_color_str]
			new_color_str =  "cwcmod:cwc_" + new_color_str + "_rn"

			new_block["Type"] = new_color_str

			return new_block

		new_world_state["BlocksInGrid"] = list(map(h, world_state["BlocksInGrid"]))

		def j(utterance):

			def get_sub(token):
				if token in color_map:
					token_substitution = color_map[token]
				else:
					token_substitution = token
				return token_substitution

			all_tokens = tokenize_utterance(utterance, lower)
			all_tokens_substituted = list(map(get_sub, all_tokens))

			return all_tokens_substituted

		new_world_state["ChatHistory"] = list(map(j, world_state["ChatHistory"]))

		return new_world_state

	new_observations_json["WorldStates"] = list(map(g, observations_json["WorldStates"]))

	# return final output

	return new_observations_json, new_config_structure

def apply_transformation(observations_json, transformation):
	"""
		Apply transformation to every world state -- every builder position and built config
	"""
	new_observations_json = {}

	new_observations_json["NumFixedViewers"] = observations_json["NumFixedViewers"]
	new_observations_json["TimeElapsed"] = observations_json["TimeElapsed"]

	def g(world_state):
		new_world_state = {}

		new_world_state["BuilderInventory"] = world_state["BuilderInventory"]
		new_world_state["ChatHistoryTokenized"] = world_state["ChatHistoryTokenized"]
		new_world_state["Timestamp"] = world_state["Timestamp"]
		new_world_state["ScreenshotPath"] = world_state["ScreenshotPath"]

		def reformat(block):
			return {
				"x": block["AbsoluteCoordinates"]["X"],
				"y": block["AbsoluteCoordinates"]["Y"],
				"z": block["AbsoluteCoordinates"]["Z"],
				"type": block["Type"]
			}
		built_config = list(map(reformat, world_state["BlocksInGrid"]))
		builder_position = world_state["BuilderPosition"]

		new_built_config, new_builder_position = spatially_transform(built_config, builder_position, transformation)

		def h(index, block): # TODO: TEST
			new_block = {}

			new_block["Type"] = block["Type"]
			new_block["PerspectiveCoordinates"] = None

			new_block["AbsoluteCoordinates"] = {
				"X": new_built_config[index]["x"],
				"Y": new_built_config[index]["y"],
				"Z": new_built_config[index]["z"]
			}

			return new_block

		new_world_state["BlocksInGrid"] = list(map(
			lambda x: h(x[0], x[1]), enumerate(world_state["BlocksInGrid"])
		))

		new_world_state["BuilderPosition"] = new_builder_position

		return new_world_state

	new_observations_json["WorldStates"] = list(map(g, observations_json["WorldStates"]))

	return new_observations_json

def spatially_transform(built_config, builder_position, transformation):
	"""
		Applies a transformation to both args and returns them
	"""

	new_built_config = generate_perturbation(
		config=built_config,
		x_target=transformation.x_target, z_target=transformation.z_target,
		rot_target=transformation.rot_target,
		gold_config=None
	).perturbed_config

	# transform builder position
	if builder_position:
		dummy_block = {
			'x': builder_position['X'],
			'y': builder_position['Y'],
			'z': builder_position['Z'],
			'type': None, # dummy value -- doesn't matter
		}

		dummy_block_perturbed = perturb_builder_position(
			dummy_block,
			transformation.x_target, transformation.z_target, transformation.rot_target
		)

		new_builder_position_yaw = builder_position["Yaw"] + transformation.rot_target

		new_builder_position = {
			"X": dummy_block_perturbed["x"],
			"Y": dummy_block_perturbed["y"],
			"Z": dummy_block_perturbed["z"],
			"Yaw": new_builder_position_yaw,
			"Pitch": builder_position["Pitch"]
		}
	else:
		new_builder_position = None

	# # NOTE: TESTING
	# if len(built_config) == 1:
	# 	print("HERE")
	# 	print(built_config)
	# 	print("\n")
	# 	print(builder_position)
	# 	print("\n\n")
	# 	print(transformation.__dict__)
	# 	print("\n\n")
	# 	print(new_built_config)
	# 	print("\n")
	# 	print(new_builder_position)
	# 	# print("\n\n")
	# 	# both sets of perspective coordinates should be same
	# 	initial = append_block_perspective_coords(builder_position, built_config[0])
	# 	final = append_block_perspective_coords(new_builder_position, new_built_config[0])
	# 	print(initial["px"], initial["py"], initial["pz"])
	# 	print(final["px"], final["py"], final["pz"])
	# 	print("\n\n")

	return (new_built_config, new_builder_position)

# UTILS

def get_gold_config(config_xml_file):
	"""
	Args:
		config_xml_file: The XML file for a gold configuration

	Returns:
		The gold config as a list of dicts -- one dict per block
	"""
	with open(config_xml_file) as f:
		all_lines = map(lambda t: t.strip(), f.readlines())

	gold_config = list(map(ET.fromstring, all_lines))

	return gold_config

def tokenize_utterance(utterance, lower):
	prefix = "<Architect>" if "Architect" in utterance.split()[0] else "<Builder>"

	utterance = utterance[len(prefix):]

	if lower:
		utterance = utterance.lower()
	utterance_tokenized, _ = tokenize(utterance)

	return [ prefix ] + utterance_tokenized

class SyntheticObservations:
	"""
		A synthetic list of observations representing one dialog
	"""
	def __init__(self, log_dir, json_content):
		self.log_dir = log_dir
		self.json_content = json_content

class SyntheticGoldConfig:
	"""
		A synthetic gold configuration
	"""
	def __init__(self, filename, xml_content):
		self.filename = filename
		self.xml_content = xml_content

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('augmentation_factor', type=int, default=2, help='number of synthetic dialogs to be generated per original dialog')
	parser.add_argument('--lower', default=False, action='store_true', help='lowercase tokens in the dataset')
	parser.add_argument('--out_data_path', default='../data/augmented/', help='where to write aug data')

	parser.add_argument('--skip_paraphrasing', default=False, action='store_true', help='')
	parser.add_argument('--skip_color_permutation', default=False, action='store_true', help='')
	parser.add_argument('--skip_spatial_transform', default=False, action='store_true', help='')

	parser.add_argument('--seed', type=int, default=1234, help='random seed')

	args = parser.parse_args()

	initialize_rngs(args.seed, torch.cuda.is_available())

	x = DataAugmentor(
		data_path = "../data/logs/",
		gold_configs_dir = "../data/gold-configurations/",
		augmentation_factor = args.augmentation_factor,
		out_data_path = args.out_data_path,
		substitutions_lexicon_file = "../data/lexicons/synonym_substitutions.json",
		lower = args.lower,
		skip_paraphrasing = args.skip_paraphrasing,
		skip_color_permutation = args.skip_color_permutation,
		skip_spatial_transform = args.skip_spatial_transform,
	)

	print('Started generating synthetic data ...')
	start_time = time.time()
	x.generate()
	end_time = time.time()
	print('DONE!')
	print('Time elapsed: %5.2fs' % (end_time - start_time) )
