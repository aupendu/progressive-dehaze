import os
import sys
import torch
from torch.autograd import Variable
import imageio
import numpy as np
import time
from skimage.transform import resize
import math
import torchvision
import model
import argparse

parser = argparse.ArgumentParser(description='Single Image Dehazing')
parser.add_argument('--datatype', type=str, help='Hazy Image type (synthetic | real)')
parser.add_argument('--modelx', default='IPUDN_MSBDN', type=str, help='Hazy Image folder name')
parser.add_argument('--testfolder', type=str, help='Hazy Image folder name')
args = parser.parse_args()


_modelh = model.Model(args.modelx)
_modelt = model.Model('Transmission')
_modela = model.Model('AtmLocal')


checkpoint = torch.load('model_dir/'+args.modelx+'.pth.tar')
_modelt.load_state_dict(checkpoint['modelt'])
_modela.load_state_dict(checkpoint['modela'])
_modelh.load_state_dict(checkpoint['modelh'])

_modelh.cuda()
_modela.cuda()
_modelt.cuda()

print(checkpoint['epoch'])


def _np2Tensor(img):
	np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
	tensor = torch.from_numpy(np_transpose).float()
	return torch.unsqueeze(tensor, 0)



def test(args):
	torch.set_grad_enabled(False)
	_modela.eval()
	_modelt.eval()
	_modelh.eval()
	hazyimages = os.listdir('testdata/'+args.testfolder+'/hazy/')

	if not os.path.exists('testdata/'+args.testfolder+'/hazy'):
		raise RuntimeError('test folder not found in testdata: testdata/'+args.testfolder+'/hazy')

	if not os.path.exists(args.modelx+'/'+args.testfolder):
		os.makedirs(args.modelx+'/'+args.testfolder)

	for i in range(len(hazyimages)):
		sys.stdout.write("Images Processed: %d/ %d  \r" % (i+1, len(hazyimages)))
		sys.stdout.flush()

		HazyImage = imageio.imread('testdata/'+args.testfolder+'/hazy/'+hazyimages[i])
		HazyImage = _np2Tensor(HazyImage)
		HazyImage = (HazyImage/255.).cuda()
		W, H = HazyImage.shape[2], HazyImage.shape[3]
		W = W - W%32
		H = H - H%32
		HazyImage = HazyImage[:, :, 0:W, 0:H]

		if args.datatype=='synthetic':
			HazeX = HazyImage
			if HazyImage.shape[2]%32!=0 or HazyImage.shape[3]%32!=0:
				raise RuntimeError('Synthetic Image size must be the multiple of 32')
		elif args.datatype=='real':
			HazeX = torch.nn.functional.interpolate(HazyImage, size=(224, 224), mode='bicubic', align_corners=False)
			HazeX = torch.clamp(HazeX, 0, 1)
		else:
			raise RuntimeError('datatype should be mentiond: synthetic or real')

		torch.cuda.empty_cache()
		with torch.no_grad():

			trans_map = _modelt(HazeX)
			atm_map = _modela(HazeX)

			atm_map = torch.unsqueeze(torch.unsqueeze(atm_map, 2), 2)
			atm_map = atm_map.expand_as(HazyImage)

			if args.datatype=='real':
				trans_map = torch.nn.functional.interpolate(trans_map, size=(HazyImage.shape[2], HazyImage.shape[3]), mode='bicubic', align_corners=False)
				trans_map = torch.clamp(trans_map, 0, 1)

			HazefreeImage = _modelh((HazyImage, trans_map, atm_map))
			HazefreeImage = torch.round(255*torch.clamp(HazefreeImage, 0, 1))

		HazefreeImage = torch.squeeze(HazefreeImage).cpu()
		HazefreeImage = np.uint8(HazefreeImage.numpy().transpose((1, 2, 0)))
		imageio.imwrite(args.modelx+'/'+args.testfolder+'/'+hazyimages[i][:-4]+'.png', HazefreeImage)
	torch.set_grad_enabled(True)


test(args)