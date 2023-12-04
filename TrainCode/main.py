import torch
import model
from option import args
from trainer import Trainer
from dataloader import Data
import numpy as np
import random
import loss


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def main():
	global model
	loader = Data(args)
	_loss = loss.Loss(args)
	if args.train_h:
		_modelh = model.Model(args, 'haz')
		_modelt = model.Model(args, 'trn')
		_modela = model.Model(args, 'atm')
		t = Trainer(args, loader, _modelt, _modela, _modelh, _loss)
	else:
		if args.train_a:
			_model = model.Model(args, 'atm')
			t = Trainer(args, loader, my_modela=_model, my_loss=_loss)
		if args.train_t:
			_model = model.Model(args, 'trn')
			t = Trainer(args, loader, my_modelt=_model, my_loss=_loss)
	while not t.terminate():
		t.train()
		t.valid()
if __name__ == '__main__':
	main()
