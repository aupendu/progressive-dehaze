import glob
import torch
import torchvision
import time
import random
import numpy as np
from torch.utils.data.dataset import Dataset
import imageio
import pickle
import scipy.io

import loss
from dataloader import Data
import model
from option import args
from common import Image_Quality_Metric


args.loss = '1*MSE'
_lossA = loss.Loss(args)
args.loss = '1*SSIM'
_lossT = loss.Loss(args)

args.loss = '1*L1+0.8*VGG2022'
args.batch_size = 2
args.iter = 6
args.test_every = 2000
args.epochs = 500
args.patch_size = 224
args.trainfolder = 'NR-Indoor+NR-Outdoor'
args.b_min = '0.2+2.0'
args.b_max = '0.8+5.0'
args.valset = 'NR_Indoor_Outdoor'
args.valfolder = 'NR-Indoor+NR-Outdoor'
args.b_minVal = '0.2+2.0'
args.b_maxVal = '0.8+5.0'
args.A_vary = 40
args.A_min = 0.3
args.A_max = 1.0
args.transmodel = 'Transmission'
args.atmmodel = 'Atmospheric'
args.hazemodel = 'IPUDN_IHaze'
args.transmodel_pt = 'model_dir/EX1_Transmission_1*SSIM_bestSSIM.pth.tar'
args.atmmodel_pt = 'model_dir/EX1_Atmospheric_1*MSE_bestMSE.pth.tar'
args.hazemodel_pt = 'model_dir/I6EX1_IPUDN_IHaze_1*L1_bestPSNR.pth.tar'

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
    if isinstance(module, torch.nn.modules.normalization.GroupNorm):
        module.eval()

quality = Image_Quality_Metric()

loader = Data(args)

loader_train = loader.loader_train
loader_valid = loader.loader_valid

_modelt = model.Model(args, 'trn')
_modela = model.Model(args, 'atm')
_modelh = model.Model(args, 'haz')

_modelt = _modelt.cuda()
_modela = _modela.cuda()
_modelh = _modelh.cuda()

checkpoint = torch.load(args.transmodel_pt)
_modelt.load_state_dict(checkpoint['model'])

checkpoint = torch.load(args.atmmodel_pt)
_modela.load_state_dict(checkpoint['model'])

checkpoint = torch.load(args.hazemodel_pt)
_modelh.load_state_dict(checkpoint['model'])


_optimt = torch.optim.Adam(_modelt.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
_optima = torch.optim.Adam(_modela.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
_optimh = torch.optim.Adam(_modelh.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)


_loss = loss.Loss(args)


bestval = 0


for epoch in range(args.epochs): 

	if (epoch+1)%100==0:

		for param_group in _optimt.param_groups:
			lr = param_group["lr"]
			param_group["lr"] = lr*0.5
		for param_group in _optima.param_groups:
			lr = param_group["lr"]
			param_group["lr"] = lr*0.5
		for param_group in _optimh.param_groups:
			lr = param_group["lr"]
			param_group["lr"] = lr*0.5
	for param_group in _optimt.param_groups:
		lr = param_group["lr"]

	_modelt.train()
	_modela.train()
	_modelh.train()
	_modelt.apply(set_bn_eval)
	_modela.apply(set_bn_eval)
	print('At Epoch: %d; Learning Rate: %f; Iteration: %d'%(epoch+1, lr, args.iter))
	print('\n')
	startepoch = time.time()
	h_loss = 0
	t_loss = 0
	a_loss = 0

	for i_batch, (GT, Hazy, Trans, Atmos) in enumerate(loader_train):

		Hazy, GT = Hazy/255.0, GT/255.0
		GT, Hazy = GT.cuda(), Hazy.cuda()
		Trans, Atmos = Trans.cuda(), Atmos.cuda()

		_optimt.zero_grad()
		_optima.zero_grad()
		_optimh.zero_grad()

		_haze = _modelt(Hazy)
		_ambt = _modela(Hazy)

		_amb = torch.unsqueeze(torch.unsqueeze(_ambt, 2), 2)
		_amb = _amb.expand_as(Hazy)

		_GT = _modelh((Hazy.cuda(), _haze, _amb, False))
		
		ilossA = _lossA(_ambt, torch.squeeze(Atmos))
		ilossT = _lossT(_haze, Trans)
		ilossX = _loss(_GT, GT)

		iloss = ilossX + ilossT + ilossA
		
		iloss.backward()

		_optimt.step()
		_optima.step()
		_optimh.step()

		h_loss += ilossX.data.cpu()
		t_loss += ilossT.data.cpu()
		a_loss += ilossA.data.cpu()

		if (i_batch+1)%100==0:
			h_loss = h_loss/100
			t_loss = t_loss/100
			a_loss = a_loss/100
			total_time = time.time()-startepoch
			startepoch = time.time()
			print('[Batch: %d] [Dehaze Loss: %f] [Trans Loss: %f] [Atmos Loss: %f] [Time: %f s]'\
				%(i_batch+1, h_loss, t_loss, a_loss, total_time))
			h_loss = 0
			t_loss = 0
			a_loss = 0


	val_psnr = 0
	_modelt.eval()
	_modela.eval()
	_modelh.eval()
	for i_batch, (GT, Hazy, _, _) in enumerate(loader_valid):

		Hazy, GT = (Hazy/255.0).cuda(), (GT/255.0).cuda()

		with torch.no_grad():
			_haze = _modelt(Hazy)
			_ambt = _modela(Hazy)
			_ambt = torch.unsqueeze(torch.unsqueeze(_ambt, 2), 2)
			_ambt = _ambt.expand_as(Hazy)
			_GT = _modelh((Hazy, _haze, _ambt, False))

		Hazefreebatch = torch.clamp(torch.round(255.0*_GT), 0, 255)
		
		val_psnr += quality.psnr(Hazefreebatch/255.0, GT, rgb_range=1)

	val_psnr = val_psnr/len(loader_valid)

	torch.save({'epoch': epoch+1, 
					'modelt': _modelt.state_dict(), 'modela': _modela.state_dict(), 'modelh': _modelh.state_dict(),
					'optimt': _optimt.state_dict(), 'optima': _optima.state_dict(), 'optimh': _optimh.state_dict(), 
					'psnr': val_psnr, 'epoch': epoch+1,},
					'model_dir/'+args.hazemodel+'_check.pth.tar')

	if (val_psnr)>bestval:
		bestepoch = epoch+1
		bestval = val_psnr
		torch.save({'epoch': epoch+1, 
					'modelt': _modelt.state_dict(), 'modela': _modela.state_dict(), 'modelh': _modelh.state_dict(),
					'optimt': _optimt.state_dict(), 'optima': _optima.state_dict(), 'optimh': _optimh.state_dict(), 
					'psnr': val_psnr, 'epoch': epoch+1,}, 
					'model_dir/'+args.hazemodel+'_best.pth.tar')
	
	print('\n\nValidation PSNR:   %.3f  (Best PSNR: %.3f at %d)'%(val_psnr, bestval, bestepoch))


