import torch
from torch import nn
from torch.autograd import Variable

class inception(nn.Module):
	def __init__(self, input_size, config):
		self.config = config
		super(inception,self).__init__()
		self.convs = nn.ModuleList()

		# Base 1*1 conv layer
		self.convs.append(nn.Sequential(
			nn.Conv2d(input_size, config[0][0],1),
			nn.BatchNorm2d(config[0][0],affine=False),
			nn.ReLU(True),
		))

		# Additional layers
		for i in range(1, len(config)):
			filt = config[i][0]
			pad = int((filt-1)/2)
			out_a = config[i][1]
			out_b = config[i][2]
			conv = nn.Sequential(
				nn.Conv2d(input_size, out_a,1),
				nn.BatchNorm2d(out_a,affine=False),
				nn.ReLU(True),
				nn.Conv2d(out_a, out_b, filt,padding=pad),
				nn.BatchNorm2d(out_b,affine=False),
				nn.ReLU(True)
				)
			self.convs.append(conv)

	def __repr__(self):
		return "inception"+str(self.config)

	def forward(self, x):
		ret = []
		for conv in (self.convs):
			ret.append(conv(x))
		return torch.cat(ret,dim=1)

class Channels1(nn.Module):
	def __init__(self):
		super(Channels1, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]])
				)
			) #EE
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
				# nn.UpsamplingNearest2d(scale_factor=2)
				nn.Upsample(scale_factor=2)
				)
			) #EEE

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels2(nn.Module):
	def __init__(self):
		super(Channels2, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]),
				inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]])
				)
			)#EF
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]),
				Channels1(),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]),
				inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]]),
				# nn.UpsamplingNearest2d(scale_factor=2)
				nn.Upsample(scale_factor=2)

				)
			)#EE1EF

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels3(nn.Module):
	def __init__(self):
		super(Channels3, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
				inception(128, [[64], [3,32,64], [5,32,64], [7,32,64]]),
				Channels2(),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]),
				inception(256, [[32], [3,32,32], [5,32,32], [7,32,32]]),
				# nn.UpsamplingNearest2d(scale_factor=2)
				nn.Upsample(scale_factor=2)

				)
			)#BD2EG
		self.list.append(
			nn.Sequential(
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
				inception(128, [[32], [3,64,32], [7,64,32], [11,64,32]])
				)
			)#BC

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels4(nn.Module):
	def __init__(self):
		super(Channels4, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
				Channels3(),
				inception(128, [[32], [3,64,32], [5,64,32], [7,64,32]]),
				inception(128, [[16], [3,32,16], [7,32,16], [11,32,16]]),
				# nn.UpsamplingNearest2d(scale_factor=2)
				nn.Upsample(scale_factor=2)

				)
			)#BB3BA
		self.list.append(
			nn.Sequential(
				inception(128, [[16], [3,64,16], [7,64,16], [11,64,16]])
				)
			)#A

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.seq = nn.Sequential(
			nn.Conv2d(1,128,7,padding=3),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			Channels4(),
			nn.Conv2d(64,3,3,padding=1)
			)

	def forward(self,x):
		# print(x.data.size())
		return self.seq(x)
