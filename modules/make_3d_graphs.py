import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
from torch_geometric.data import (Data, InMemoryDataset, download_url, extract_zip)
import torch.nn.functional as F
from math import sqrt

def distance(a, b):
    return sqrt((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2)

from . import scaling as scl
atom_types={'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Br':35}
str_types = {'1':'H', '6':'C', '7':'N', '8':'O', '9':'F', '14':'Si', '15':'P','16':'S', '17':'Cl', '35':'Br'} 

def make_graphs(data):
    
    print('scaling..')
    scl_dict={}
    data['normalized_shift']=0

    for atom_type in tqdm(atom_types):
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

    failed = []
    mols = []
    for i in range(len(data)):
        mols.append(data.iloc[i]['molecule_name'])

    p = np.array(mols)
    p = np.unique(p)
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Si': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br': 9}
    bond_types = [0, 1, 2, 3]
    data_list=[]

    for i, mol in enumerate(tqdm(p)):
        

        y = []
        mol_df = data[data['molecule_name']==mol]
        N = len(mol_df)
        name = mol
        pos = [] 
        
        type_idx = []
        atomic_number = []
        target = []
        c=0
        row=[]
        col=[]
        distance_list=[]
        bond_type = []
        for index, atom in enumerate(range(N)):
            pos=[]
            pos.append(mol_df.iloc[atom]['x'])
            pos.append(mol_df.iloc[atom]['y'])
            pos.append(mol_df.iloc[atom]['z'])
            d=0
            conn = mol_df.iloc[atom]['conn']

            for second_index, other_atom in enumerate(range(N)):
                row+=[index]
                col+=[second_index]

                s_pos=[]
                s_pos.append(mol_df.iloc[other_atom]['x'])
                s_pos.append(mol_df.iloc[other_atom]['y'])
                s_pos.append(mol_df.iloc[other_atom]['z'])
                            
                distance_list.append(distance([float(x) for x in pos], [float(y) for y in s_pos]))    
                bond_type.append(int(conn[other_atom]))

                d+=1
            c+=1
                
            type_idx.append(types[mol_df.iloc[atom]['typestr']])
            atomic_number.append(float(mol_df.iloc[atom]['typeint']))
            target.append(mol_df.iloc[atom]['normalized_shift'])
            
        edge_index = torch.tensor([row, col], dtype=torch.long)
        dist = scl.normalize(distance_list, min_max=True)

        bond_type = F.one_hot(torch.tensor(bond_type), num_classes=len(bond_types))
        dist = torch.tensor([dist], dtype=torch.float32, requires_grad=True).t().contiguous()
        edge_attr = torch.cat([bond_type.to(torch.float32), dist], dim=-1)
            
        y =  torch.tensor(np.array(target), dtype=torch.float32).unsqueeze(dim=1)

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
        x2 = torch.tensor([atomic_number], dtype=torch.float32, requires_grad=True).t().contiguous()
        x = torch.cat([x1.to(torch.float32), x2], dim=-1)

        #x = torch.tensor(np.array(type_idx), dtype=torch.float32).unsqueeze(1)

        graph =Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                     y=y, idx=i)

        data_list.append(graph)

        

    return data_list, scl_dict, data, failed

