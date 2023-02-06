import torch
from tqdm import tqdm

def do_train(model, optimizer, num_layers, criterion, train_loader, progress=True):
	
	for batch in train_loader:
	
		model.train()
		optimizer.zero_grad()
		out = model(num_layers, batch.x.t()[:10].t(), batch.edge_index, batch.edge_attr)
		loss = criterion(out, batch.y)
		loss.backward()
		optimizer.step()
	
	return loss
