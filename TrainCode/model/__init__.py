import torch.nn as nn
from importlib import import_module

class Model(nn.Module):
    def __init__(self, args, modelname):
        super(Model, self).__init__()
        
        print('Making model...')
        
        if modelname=='trn': module = import_module('model.' + args.transmodel)
        if modelname=='atm': module = import_module('model.' + args.atmmodel)
        if modelname=='haz': module = import_module('model.' + args.hazemodel)
        self.model = module.make_model(args)

    def forward(self, x):
        return self.model(x)
