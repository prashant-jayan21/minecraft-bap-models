import sys, os, argparse, numpy as np, json, csv
from glob import glob
from collections import defaultdict

sys.path.append("../")

from utils import get_config_params, Logger, write_commit_hashes

def main(args):
	""" Gets all eval.txt files for models in a given directory and compiles them into a single document (as well as single csv).
	Also prints information on the best-performing model. """

	# iterate over model directories that have successfully been trained and evaluated
	all_csv_content = []

	for config_file in glob(args.models_dir+'/**/config.txt', recursive=True):
		# print('Accumulating evals for model:', '/'.join(config_file.split('/')[:-1]))
		args_sfx = '-'+args.model_iteration+'-'+args.split+('' if not args.development_mode else '-development_mode')
		args_sfx += '-beam_'+str(args.beam_size)
		args_sfx += ('' if not args.masked_decoding else '-masked')

		if not os.path.exists(config_file.replace('config.txt','generated_sentences'+args_sfx+'.txt')):
			print(config_file)
			continue

		# get config
		_, config_params = get_config_params(config_file)
		csv_content = config_params

		# get model location
		model_path = ('/'.join(os.path.abspath(config_file).split("/")[:-1]))
		csv_content["model_path"] = model_path

		# get results
		results = get_results_2(config_file.replace('config.txt','generated_sentences'+args_sfx+'.txt'))
		for key in results:
			csv_content[key] = results[key]

		results = get_results(config_file.replace('config.txt','eval-'+("best" if args.model_iteration == "best_by_loss" else "best-action_f1")+'.txt'))
		for key in results:
			csv_content[key] = results[key]

		all_csv_content.append(csv_content)

	keys = all_csv_content[0].keys()
	with open(os.path.join(args.models_dir, 'cumulative_evals'+args_sfx+'.csv'), 'w') as output_file:
		dict_writer = csv.DictWriter(output_file, keys)
		dict_writer.writeheader()
		dict_writer.writerows(all_csv_content)

	print("\nSaving git commit hashes ...\n")
	write_commit_hashes("../..", args.models_dir, filepath_modifier="_compile_results")

def get_results(eval_file):
	lines = [line.rstrip('\n') for line in open(eval_file)]

	results = {}
	for line in lines:
		pair = line.split(" ")
		results[pair[0]] = float(pair[1])

	return results

def get_results_2(eval_file):
	lines = []
	for line in open(eval_file):
		if line == '\n':
			break
		else:
			lines.append(line.rstrip('\n'))

	results = {}
	for line in lines:
		pair = line.split(" ")
		results[pair[0]] = float(pair[1])

	return results

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('models_dir', type=str, help='path for models saved on a specific date, each containing their own config & eval files')
	parser.add_argument('--model_iteration', default='best_by_loss', help='iteration of model to be evaluated: best, final, best_by_loss, best_by_f1')
	parser.add_argument('--beam_size', type=int, default=1, help='beam size for beam search decoding')
	parser.add_argument("--development_mode", default=False, action="store_true", help="Whether or not to run in development mode, i.e., with less data")
	parser.add_argument('--masked_decoding', default=False, action='store_true', help='whether or not to use masked decoding to mask out infeasible actions')
	parser.add_argument('--split', default='val', help='data split from which sentences should be generated')
	args = parser.parse_args()
	main(args)
