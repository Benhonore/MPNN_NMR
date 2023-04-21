from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

import numpy as np
import copy

def mae(x, y):
	sum(abs(x-y))/len(x)

def setup_gaussian(param_ranges, kappa, xi):
	
	pbounds = {}
	for param in param_ranges.keys():
		pbounds[param] =  (0, 1)
		
		print(pbounds)

	optimizer = BayesianOptimization(
		f=None,
		pbounds=pbounds,
		verbose=1,
		random_state=None)

	utility = UtilityFunction(kind='ucb', kappa=kappa, xi=xi)

	return optimizer, utility

def gaussian_iteration(opt, util, model, tr_graphs, te_graphs, tr_scl_dict, param_ranges={}):
	
	next_point_to_probe = opt.suggest(util)
	params = {}

	for param in param_ranges.keys():
		diff = param_ranges[param]['max'] - param_ranges[param]['min']
		params[param] = (next_point_to_probe[param]*diff) + param_ranges[param]['min']
		
		if param_ranges[param]['log']:
			params[param] = 10**params[param]
	
	model.params = copy.copy(params)
	model.check_params()

	print(model.params)

	model.params['batch_size'] = int(model.params['batch_size'])
	model.params['embedding_size'] = int(model.params['embedding_size'])
	model.params['num_layers'] = int(model.params['num_layers'])
	
	print(model.params)

	model.tr_input = model.get_input(tr_graphs)
	losses = model.train(model.tr_input)

	model.te_input = model.get_input(te_graphs)
	df = model.predict(model.te_input)

	carbon_score = mae(df[df['typestr']=='C']['shift'], df[df['typestr']=='C']['predicted_shift'])
	print(f'carbon score: {carbon_score}')

	c_range = tr_scl_dict['C']['max'] - tr_scl_dict['C']['min']
	scl_c_score = carbon_score/c_range

	proton_score = mae(df[df['typestr']=='H']['shift'], df[df['typestr']=='H']['predicted_shift'])
	print(f'proton_score: {proton_score}')
	p_range = tr_scl_dict['H']['max'] - tr_scl_dict['H']['min']
	scl_p_score = proton_score/p_range

	overall_score = scl_c_score + scl_p_score

	overall_score
		
