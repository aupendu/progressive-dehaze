import os
import glob
import torch
import random
import numpy as np
import imageio
import pickle
import scipy.io
from os import path


"""
args.trainfolder = NR-Outdoor+NR-Indoor
args.b_min = 2.0+0.2
args.b_max = 5.0+0.8
args.A_min = 0.3
args.A_max = 1.0
args.A_vary = 40
args.remove_boarder = 10
args.patch_size = 224

args.batch_size = 
args.test_every = 
"""


class CustomDataset(Dataset):
	def __init__(self, args, itype):
		self.args = args
		self.itype = itype
		if self.itype=='train':
			self.paths = args.trainfolder.split('+')
			self.b_min = args.b_min.split('+')
			self.b_max = args.b_max.split('+')
		self.img_paths, self.dep_paths, self.file_types = [], [], []
		for f in range(len(self.paths)):
			self.fpath = os.path.join('your data path', itype, self.paths[f])
			self.files = os.listdir(os.path.join(self.fpath, 'clean'))
			self.path_bin = os.path.join('your data path', 'bin', itype, self.paths[f])
			os.makedirs(self.path_bin, exist_ok=True)
			imgpaths, deppaths, filetypes = self._savetrain(self.paths[f])
			self.img_paths.extend(imgpaths)
			self.dep_paths.extend(deppaths)
			self.file_types.extend(filetypes)
		self.repeat = (args.batch_size*args.test_every)//len(self.img_paths)
		if self.repeat==0: self.repeat=1


	def _savetrain(self, f_type):
		path_images = os.path.join(self.path_bin, 'clean')
		os.makedirs(path_images, exist_ok=True)
		path_depths = os.path.join(self.path_bin, 'depth')
		os.makedirs(path_depths, exist_ok=True)
		for i in range(len(self.files)):
			if not os.path.exists(path_images+'/'+self.files[i][:-4]+'.pt'):
				print('Making file:  '+self.files[i])
				_image = imageio.imread(self.fpath+'/clean/'+self.files[i], pilmode="RGB")
				_depth = scipy.io.loadmat(self.fpath+'/depth/'+self.files[i][:-4]+'.mat')
				_depth = _depth['depth']
				if f_type=='NR-Outdoor': 
					_depth = 1 - np.float32(_depth)
				if f_type=='NR-Indoor':
					_depth = _depth[self.args.remove_boarder:-self.args.remove_boarder, self.args.remove_boarder:-self.args.remove_boarder]
					_image = _image[self.args.remove_boarder:-self.args.remove_boarder, self.args.remove_boarder:-self.args.remove_boarder, :]
				with open(path_images+'/'+self.files[i][:-4]+'.pt', 'wb') as _f:
					pickle.dump(_image, _f)
				with open(path_depths+'/'+self.files[i][:-4]+'.pt', 'wb') as _f:
					pickle.dump(_depth, _f)
		file_type = [f_type] * len(self.files)
		return glob.glob(path_images+'/*.pt'), glob.glob(path_depths+'/*.pt'), file_type


	
	def _crop(self, iImage, iDepth, psize, ptype):
		ih, iw = iImage.shape[:2]
		x0 = random.randrange(0, ih-psize+1)
		x1 = random.randrange(0, iw-psize+1)
		iDepth = iDepth[x0:x0+psize, x1:x1+psize]
		iImage = iImage[x0:x0+psize, x1:x1+psize, :]
		for l in range(len(self.paths)):
			if ptype==self.paths[l]:
				iImage, Himage, ttx, A = self._addhaze(iImage, iDepth, l)
		return self._np2Tensor(iImage), self._np2Tensor(Himage), self._np2Tensor(ttx), self._np2Tensor(A)

	def _addhaze(self, Ibatch, iDepth, l):
		ttx = np.exp(-self._findrand(self.b_min[l], self.b_max[l])*iDepth)
		ttx = np.expand_dims(ttx, axis=2)
		A = self._findrand(self.args.A_min, self.args.A_max)
		A0 = A*self._findrand(1-self.args.A_vary/100, 1+self.args.A_vary/100)
		A1 = A*self._findrand(1-self.args.A_vary/100, 1+self.args.A_vary/100)
		A2 = A*self._findrand(1-self.args.A_vary/100, 1+self.args.A_vary/100)
		A = np.array([A0, A1, A2])
		A = np.clip(A, float(self.args.A_min), float(self.args.A_max))
		A = np.expand_dims(np.expand_dims(A, axis=0), axis=0)
		Ibatch, ttx = self._augment(Ibatch, ttx)

		Hbatch = (np.float32(Ibatch)/255.0)*ttx + (1-ttx)*A
		Hbatch = np.clip(Hbatch, 0.0, 1.0)

		Hbatch = np.uint8(np.round(Hbatch*255.0))
		return Ibatch, Hbatch, ttx, A

	def _findrand(self, _min, _max):
		tmin = round(10000*float(_min))
		tmax = round(10000*float(_max))
		num = random.randint(tmin, tmax)
		return float(num)/10000

	def _augment(self, iImg, ttx, is_aug=True):
		hflip = is_aug and random.random() < 0.5
		vflip = is_aug and random.random() < 0.5
		rot90 = is_aug and random.random() < 0.5
		if hflip: 
			iImg = iImg[:, ::-1, :]
			ttx  =  ttx[:, ::-1, :]
		if vflip: 
			iImg = iImg[::-1, :, :]
			ttx  =  ttx[::-1, :, :]
		if rot90: 
			iImg = iImg.transpose(1, 0, 2)
			ttx  =  ttx.transpose(1, 0, 2)
		return iImg, ttx

	def _np2Tensor(self, img):
		np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
		tensor = torch.from_numpy(np_transpose).float()
		return tensor

	def __getitem__(self, index):

		index = self._get_index(index)
		with open(self.img_paths[index], 'rb') as _f:
			IMG = pickle.load(_f)
		with open(self.dep_paths[index], 'rb') as _f:
			DEP = pickle.load(_f)
		Ibatch, Hbatch, Tbatch, Abatch = self._crop(IMG, DEP, self.args.patch_size, self.file_types[index])

		return Ibatch, Hbatch, Tbatch, Abatch


	def __len__(self): 
		len(self.img_paths)*self.repeat

	def _get_index(self, index):
		return index % len(self.img_paths)

class Data:
	def __init__(self, args):
		train_dataset = CustomDataset(args, 'train')
		self.loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
