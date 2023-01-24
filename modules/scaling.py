import pandas as pd
import numpy as np

def normalize(data, min_max=False):
	if min_max:
		return (data - np.min(data)) / (np.max(data) - np.min(data))
	else:
		return (data - np.mean(data)) / np.std(data)

def denormalize(data, target_dict, min_max=False):
	if min_max:
		return (data*(float(target_dict['max'])-float(target_dict['min']))) + float(target_dict['min'])

	else:
		return (data*float(target_dict['std'])) + float(target_dict['mean'])

def make_scl_dict(data):
	target_dict = {'min': np.min(data), 'max' : np.max(data), 'mean' : np.mean(data), 'std' : np.std(data)}	
	return target_dict


def add_scaling(dataframe, min_max =False, atom_types=['H', 'C', 'N', 'O', 'F']):
	df = pd.read_pickle(dataframe)
	scl_dict = {}
	for atom_type in atom_types:
		ret = list(df[df['typestr']==atom_type]['shift'])
		if len(ret) == 0:
			continue
		scl_ret = list(normalize(ret))
		scl_dict[atom_type] = make_scl_dict(ret)
		c=-1
		for i in range(len(df)):
			if df.iloc[i]['typestr'] == atom_type:
				c+=1
				df.at[i, 'scaled_shift'] = scl_ret[c]
	return df, scl_dict

def assign_preds(preds, dataframe, scl_dict, atom_types=['H', 'C', 'N', 'O', 'F']):
	df=dataframe
	if len(df) != len(preds):
		print('dataframe and predictions do not match in length')
	else:
		pred_list = []
		for i in preds:
			pred_list.append(float(i))
		
		df['unscaled_predicted_shift'] = pred_list
	
	for atom_type in atom_types:
		ret = list(df[df['typestr']==atom_type]['unscaled_predicted_shift'])  	
		descl_ret = list(denormalize(ret, scl_dict[atom_type]))
		c=-1
		for i in range(len(df)):
			if df.iloc[i]['typestr']==atom_type:
				c+=1
				df.at[i, 'scaled_shift'] = descl_ret[c]
	return df

	

