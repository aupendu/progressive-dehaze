import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

def make_model():
	return IPUDN()

class ResBlock(nn.Module):
	def __init__(self):
		super(ResBlock, self).__init__()
		self.conv0 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(True))
		self.conv1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(True))
	def forward(self, x):
		fea = self.conv0(x)
		fea = self.conv1(fea)
		return x + fea

class IPUDN(nn.Module):
	def __init__(self):
		super(IPUDN, self).__init__()
		self.iteration = 6

		self.conv0 = nn.Sequential(nn.Conv2d(3+1+3+3+1+3, 32, 3, 1, 1), nn.ReLU(True))
		res_branch = []
		for i in range(6):
			res_branch.append(ResBlock())
		self.res_branch = nn.Sequential(*res_branch)
		self.conv1 = nn.Conv2d(32, 3, 3, 1, 1)

		self.conv_i = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
		self.conv_f = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
		self.conv_g = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
		self.conv_o = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

		self.trans_correction = nn.Sequential(
			nn.Conv2d(3+1+3+1, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 1, 3, 1, 1),
			nn.Tanh()
			)

		self.amb_correction = nn.Sequential(
			nn.Conv2d(3+3+3+3, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.PReLU(),
			nn.Conv2d(32, 3, 3, 1, 1),
			nn.Tanh(),
			nn.AdaptiveAvgPool2d(1)
			)

	def forward(self, x):
		# x[0]: img; x[1]: trans; x[2]: A
		img = x[0]
		trans = x[1]
		amb = x[2]

		batch_size, row, col = img.size(0), img.size(2), img.size(3)
		x = img
		xt = trans
		xa = amb
		h = Variable(torch.zeros(batch_size, 32, row, col).cuda(img.get_device()))
		c = Variable(torch.zeros(batch_size, 32, row, col).cuda(img.get_device()))
		for i in range(self.iteration):
			x = torch.cat((img, trans, amb, x, xt, xa), 1)
			x = self.conv0(x)
			x = torch.cat((x, h), 1)
			i = self.conv_i(x)
			f = self.conv_f(x)
			g = self.conv_g(x)
			o = self.conv_o(x)
			c = f * c + i * g
			h = o * torch.tanh(c)
			x = h
			x = self.conv1(self.res_branch(x))

			dt = self.trans_correction(torch.cat((img, trans, x, xt), 1))
			xt = xt+dt
			da = self.amb_correction(torch.cat((img, amb, x, xa), 1))
			da = da.expand_as(img)
			xa = xa+da
		return x



