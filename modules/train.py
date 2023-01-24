import torch
from tqdm import tqdm

def do_train(model, optimizer, criterion, train_loader, progress=True):
	
	for batch in train_loader:
	
		model.train()
		optimizer.zero_grad()
		out = model(batch.x, batch.edge_index)
		loss = criterion(out, batch.y)
		loss.backward()
		optimizer.step()
	
	return loss
