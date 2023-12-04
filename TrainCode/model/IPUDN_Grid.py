import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def make_model(args, parent=False):
	return IPUDN(args)

class MeanShift(nn.Conv2d):
	def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
		super(MeanShift, self).__init__(3, 3, kernel_size=1)
		std = torch.Tensor(rgb_std)
		self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
		self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
		for p in self.parameters():
			p.requires_grad = False



# --- Build dense --- #
class MakeDense(nn.Module):
	def __init__(self, in_channels, growth_rate, kernel_size=3):
		super(MakeDense, self).__init__()
		self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

	def forward(self, x):
		out = F.relu(self.conv(x))
		out = torch.cat((x, out), 1)
		return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
	def __init__(self, in_channels, num_dense_layer, growth_rate):
		"""
		:param in_channels: input channel size
		:param num_dense_layer: the number of RDB layers
		:param growth_rate: growth_rate
		"""
		super(RDB, self).__init__()
		_in_channels = in_channels
		modules = []
		for i in range(num_dense_layer):
			modules.append(MakeDense(_in_channels, growth_rate))
			_in_channels += growth_rate
		self.residual_dense_layers = nn.Sequential(*modules)
		self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

	def forward(self, x):
		out = self.residual_dense_layers(x)
		out = self.conv_1x1(out)
		out = out + x
		return out



# --- Downsampling block in GridDehazeNet  --- #
class DownSample(nn.Module):
	def __init__(self, in_channels, kernel_size=3, stride=2):
		super(DownSample, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
		self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.relu(self.conv2(out))
		return out


# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
	def __init__(self, in_channels, kernel_size=3, stride=2):
		super(UpSample, self).__init__()
		self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
		self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

	def forward(self, x, output_size):
		out = F.relu(self.deconv(x, output_size=output_size))
		out = F.relu(self.conv(out))
		return out


# --- Main model  --- #
class GridDehazeNet(nn.Module):
	def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
		super(GridDehazeNet, self).__init__()
		self.rdb_module = nn.ModuleDict()
		self.upsample_module = nn.ModuleDict()
		self.downsample_module = nn.ModuleDict()
		self.height = height
		self.width = width
		self.stride = stride
		self.depth_rate = depth_rate
		self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
		self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
		self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

		

		rdb_in_channels = depth_rate
		for i in range(height):
			for j in range(width - 1):
				self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
			rdb_in_channels *= stride

		_in_channels = depth_rate
		for i in range(height - 1):
			for j in range(width // 2):
				self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
			_in_channels *= stride

		for i in range(height - 2, -1, -1):
			for j in range(width // 2, width):
				self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
			_in_channels //= stride

	def forward(self, inp):
		#x = self.sub_mean(x)
		#inp = self.conv_in(x)

		x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
		i, j = 0, 0

		x_index[0][0] = self.rdb_in(inp)

		for j in range(1, self.width // 2):
			x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])

		for i in range(1, self.height):
			x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

		for i in range(1, self.height):
			for j in range(1, self.width // 2):
				channel_num = int(2**(i-1)*self.stride*self.depth_rate)
				x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
								self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])

		x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
		k = j

		for j in range(self.width // 2 + 1, self.width):
			x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])

		for i in range(self.height - 2, -1, -1):
			channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
			x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
							  self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())

		for i in range(self.height - 2, -1, -1):
			for j in range(self.width // 2 + 1, self.width):
				channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
				x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
								self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())

		out = self.rdb_out(x_index[i][j])
		#out = F.relu(self.conv_out(out))

		return out

class IPUDN(nn.Module):
	def __init__(self, args):
		super(IPUDN, self).__init__()
		self.iteration = args.iter

		self.conv0 = nn.Sequential(nn.Conv2d(3+1+3+3+1+3, 16, 3, 1, 1), nn.ReLU(True))
		self.grid = GridDehazeNet()
		self.conv1 = nn.Conv2d(16, 3, 3, 1, 1)

		self.conv_i = nn.Sequential(nn.Conv2d(16 + 16, 16, 3, 1, 1), nn.Sigmoid())
		self.conv_f = nn.Sequential(nn.Conv2d(16 + 16, 16, 3, 1, 1), nn.Sigmoid())
		self.conv_g = nn.Sequential(nn.Conv2d(16 + 16, 16, 3, 1, 1), nn.Tanh())
		self.conv_o = nn.Sequential(nn.Conv2d(16 + 16, 16, 3, 1, 1), nn.Sigmoid())

		vgg_mean = (0.5, 0.5, 0.5)
		vgg_std = (0.5, 0.5, 0.5)
		self.sub_mean = MeanShift(1.0, vgg_mean, vgg_std)

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
		return_both = x[3]

		img = self.sub_mean(img)

		batch_size, row, col = img.size(0), img.size(2), img.size(3)
		x = img
		xt = trans
		xa = amb
		h = Variable(torch.zeros(batch_size, 16, row, col).cuda(img.get_device()))
		c = Variable(torch.zeros(batch_size, 16, row, col).cuda(img.get_device()))

		for k in range(self.iteration):
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
			x = F.relu(self.conv1(self.grid(x)))

			if k<self.iteration-1:
				x = self.sub_mean(x)

				dt = self.trans_correction(torch.cat((img, trans, x, xt), 1))
				xt = xt+dt
				da = self.amb_correction(torch.cat((img, amb, x, xa), 1))
				da = da.expand_as(img)
				xa = xa+da
			
		return x



