import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from modules import scaling as scl
from modules.train import do_train
from modules.make_3d_graphs import make_graphs
from modules.NN_CONV import NNConv

from torch_geometric.loader import DataLoader


def NNConv_model():
	def __init__(self, id='GCNmodel', model_params={}):
		
		self.id = id
		self.params = model_params
		
	def check_params(self):
		if 'epochs' not in self.params.keys():
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
		if 'criterion' not in self.params.keys():
			self.params['criterion'] = torch.nn.MSELoss()

	def get_input(self, dataframe=pd.DataFrame())
		graphs, scl_dict, dataframe = make_graphs(dataframe)
		train_loader = DataLoader(graphs, batch_size=self.params['batch_size'])
		
		return train_loader

	def init_model(self):
		self.check_params()
		self.model = GCN(1, self.params['embedding_size'], self.params['dropout'])
		self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])

	def train(self, train_loader=[]):
		self.check_params()
		self.init_model()
		
		for epoch in tqdm(range(self.params['epochs'])):
			loss = do_train(self.model, self.opt, self.params['criterion'], train_loader)
			if epoch%10 == :
				print('epoch {epoch} | loss {loss}')

	def predict(self, test_loader):
		df=pd.DataFrame()
		typestr=[]
		true=[]
		preds=[]
		with torch.no_grad():
			for batch in test_loader:
				for molecule in range(len(batch.idx)):
					for atom in batch[molecule].x:
						typestr.append(str_types[str(int((atom[-1].detach().numpy())))])
				for i in (list(batch.y[:,0].detach().numpy())):
					true.append(float(i))
				pred = self.model(batch.x, batch.edge_index)
				for i in (list((pred[:,0].detach().numpy()))):
					preds.append(float(i))

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
					df.at[i, 'Shift'] = descaled_vals[c]
					c+=1

			values=[]
			for i in range(len(df)):
				if df.iloc[i]['typestr']==atom_type:
					values.append(float(df.iloc[i]['normalized_prediction']))
			predvalues = np.array(values)
			if len(predvalues)==0:
				continue
			descaled_vals=scl.denormalize(predvalues, te_scl_dict[atom_type])
			c=0
			for i in range(len(df)):
				if df.iloc[i]['typestr']==atom_type:
					df.at[i, 'Predicted Shift'] = descaled_vals[c]
					c+=1
		return df

	def save_model(self, filename):
		torch.save(self.model.state_dict(), f'{self.id}_model.pkl')
 
