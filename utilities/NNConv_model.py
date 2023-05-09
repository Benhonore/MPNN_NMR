import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from .modules import scaling as scl
from .modules.train import do_train
from .modules.make_3d_graphs import make_graphs
from .modules.NN_CONV import NNCONV

from torch_geometric.loader import DataLoader

atom_types={'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Br':35}
#str_types = {'1':'H', '6':'C', '7':'N', '8':'O', '9':'F', '14':'Si', '15':'P','16':'S', '17':'Cl', '35':'Br'} 
#str_types = {'0':'H', '1':'C', '2':'N', '3':'O', '4':'F'}
str_types = {'1':'H', '6':'C', '7':'N', '8':'O', '9':'F', '14':'Si', '15':'P','16':'S', '17':'Cl', '35':'Br'}


class NNConv_model():
	def __init__(self, id='NNConvmodel', model_params={}):
		
		self.id = id
		self.params = model_params
		
	def check_params(self):
		if 'tr_epochs' not in self.params.keys():
			self.params['tr_epochs'] = 100
		if 'batch_size' not in self.params.keys():
			self.params['batch_size'] = 2
		if 'dropout' not in self.params.keys():
			self.params['dropout'] = 0.5
		if 'learning_rate' not in self.params.keys():
			self.params['learning_rate'] = 0.007
		if 'weight_decay' not in self.params.keys():
			self.params['weight_decay'] = 5e-4
		if 'embedding_size' not in self.params.keys():
			self.params['embedding_size'] = 20
		if 'num_layers' not in self.params.keys():
			self.params['num_layers'] = 4
		if 'criterion' not in self.params.keys():
			self.params['criterion'] = torch.nn.MSELoss()

	def get_input(self, graphs):
		train_loader = DataLoader(graphs, batch_size=self.params['batch_size'])
		
		return train_loader

	def init_model(self):
		self.check_params()
		self.model = NNCONV(self.params['embedding_size'], self.params['dropout'])
		self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])

	def train(self, train_loader=[]):
		self.check_params()
		self.init_model()
		losses=[]
		
		for epoch in tqdm(range(self.params['tr_epochs'])):
			loss = do_train(self.model, self.opt, self.params['num_layers'], self.params['criterion'], train_loader)
			losses.append(loss.detach().numpy())
			if epoch%10 == 0:
				print(f'epoch {epoch} | loss {loss}')
		return losses

	def predict(self, test_loader, te_scl_dict, tr_scl_dict,  ref_df):
		df=pd.DataFrame()
		typestr=[]
		true=[]
		preds=[]
		molnames=[]
		c=0
		with torch.no_grad():
			for batch in test_loader:
				for molecule in range(len(batch.idx)):
					molname=list(ref_df['molecule_name'])[c]
					c+=1
					for atom in batch[molecule].x:
						molnames.append(molname)
						typestr.append(str_types[str(int((atom[-1].detach().numpy())))])
				for i in (list(batch.y[:,0].detach().numpy())):
					true.append(float(i))
				pred = self.model(self.params['num_layers'], batch.x.t()[:10].t(), batch.edge_index, batch.edge_attr)
				for i in (list((pred[:,0].detach().numpy()))):
					preds.append(float(i))
		
		df['molecule_name']=molnames
		df['typestr']=typestr
		df['normalized_shift'] = true
		df['normalized_prediction']=preds
		df['shift'] = 0
		df['predicted_shift'] = 0
		
		
		for atom_type in atom_types:
			values=[]
			for i in range(len(df)):
				if df.iloc[i]['typestr']==atom_type:
					values.append(float(df.iloc[i]['normalized_shift']))
			truevalues = np.array(values)
			if len(truevalues)==0:
				continue
			descaled_vals=scl.denormalize(truevalues, te_scl_dict[atom_type])
			c=0
			for i in range(len(df)):
				if df.iloc[i]['typestr']==atom_type:
					df.at[i, 'shift'] = descaled_vals[c]
					c+=1

			values=[]
			for i in range(len(df)):
				if df.iloc[i]['typestr']==atom_type:
					values.append(float(df.iloc[i]['normalized_prediction']))
			predvalues = np.array(values)
			if len(predvalues)==0:
				continue
			descaled_vals=scl.denormalize(predvalues, tr_scl_dict[atom_type])
			c=0
			for i in range(len(df)):
				if df.iloc[i]['typestr']==atom_type:
					df.at[i, 'predicted_shift'] = descaled_vals[c]
					c+=1
		return df

	def load_model(self, filename):
		checkpoint = torch.load(filename)
		self.params = checkpoint['params']
		self.init_model()
		self.model.load_state_dict(checkpoint['model'])
		self.opt.load_state_dict(checkpoint['opt'])


	def save_model(self, epoch):
		torch.save({'model':self.model.state_dict(), 'opt': self.opt.state_dict(), 'epoch': epoch, 'params': self.params}, f'{self.id}.pkl')
 
