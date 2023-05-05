import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch_geometric.data import Data

from . import scaling as scl
atom_types={'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Br':35}
str_types = {'1':'H', '6':'C', '7':'N', '8':'O', '9':'F', '14':'Si', '15':'P','16':'S', '17':'Cl', '35':'Br'} 


def make_graphs(data):
	
	print('scaling..')
	scl_dict={}
	data['normalized_shift']=0
	
	for atom_type in atom_types:
		vals=[]
		for i in range(len(data)):
			if data.iloc[i]['typestr']==atom_type:
				vals.append(data.iloc[i]['shift'])
		if len(vals)==0:
			continue
		scl_dict[atom_type] = scl.make_scl_dict(vals)
		scaled_vals = scl.normalize(vals)
		c=0
		for i in range(len(data)):
			if data.iloc[i]['typestr']==atom_type:
				data.at[i, 'normalized_shift'] = scaled_vals[c]
				c+=1	
		
	indexes = np.unique(data['molecule_name'], return_index=True)[1]
	p=[data['molecule_name'][index] for index in sorted(indexes)]	
	
	types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Si': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br': 9}
	
	data_list=[]
	
	for i, mol in enumerate(tqdm(p)):
		mol_df = data[data['molecule_name']==mol]
		pos = [] 
		pos.append(mol_df['x'])
		pos.append(mol_df['y'])
		pos.append(mol_df['z'])
		pos = np.array(pos)
		pos = torch.tensor(pos)
		pos = pos.t()
	
		type_idx = []
		atomic_number = []
		target = []
		row=[]
		col=[]
		edge_type=[]

		for atom in range(len(mol_df)):
			type_idx.append(types[mol_df.iloc[atom]['typestr']])
			atomic_number.append(float(mol_df.iloc[atom]['typeint']))
			target.append(mol_df.iloc[atom]['normalized_shift'])
			conn = mol_df.iloc[atom]['conn']
			
			for atom2, element in enumerate(conn):
				if element == 0:
					continue
				else:
					row+=[atom]
					col+=[atom2]
					edge_type+=[element-1]
			
		y =  torch.tensor(target, dtype=torch.float32).unsqueeze(dim=1)

		edge_type=torch.tensor(edge_type, dtype=torch.long)
		edge_index = torch.tensor([row, col], dtype=torch.long)
		edge_attr = F.one_hot(edge_type, num_classes=4).to(torch.float32)
		perm = (edge_index[0] * len(mol_df) + edge_index[1]).argsort()
		edge_type = edge_type[perm]
		edge_attr = edge_attr[perm]

		x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
		x2 = torch.tensor([atomic_number], dtype=torch.float32, requires_grad=True).t().contiguous()
		x = torch.cat([x1.to(torch.float32), x2], dim=-1)
		
		graph =	Data(x=x, pos=pos, edge_index=edge_index,
                	edge_attr=edge_attr, y=y,name=mol, idx=i)
	

		data_list.append(graph)

	ref_df = pd.DataFrame({'molecule_name':p, 'index':range(len(p))})

	return data_list, scl_dict, ref_df
