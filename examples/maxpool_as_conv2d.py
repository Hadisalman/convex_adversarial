import torch
import torch.nn as nn

from IPython import embed


class max_2x2(nn.Module):
	"""docstring for max_2x2"""
	def __init__(self):
		super(max_2x2, self).__init__()
		self.B = nn.Linear(3,1)
		self.B.weight.data = torch.Tensor([[1,1,-1]])
		self.B.bias.data = 0*self.B.bias.data

		self.AD = nn.Linear(6,3)
		self.AD.weight.data = torch.Tensor([[ 1.,  1., -1., -1., -1.,  1.],
											[ 0.,  0.,  0.,  1.,  1., -1.],
											[ 0.,  0.,  0., -1., -1.,  1.]])
		self.AD.bias.data = 0*self.AD.bias.data		

		self.C = nn.Linear(4,6)
		self.C.weight.data = torch.Tensor([[1, -1, 0, 0],
										   [0, 1, 0, 0],
										   [0, -1, 0, 0],
										   [0, 0, 1, -1],
										   [0, 0, 0, 1],
										   [0, 0, 0, -1]])
		self.C.bias.data = 0*self.C.bias.data

		self.C = nn.conv2d(1,6)

		# Make the parameters of this layer non-learnable
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		out = nn.Sequential(*[
			Flatten(),
			])		
		return out(x)


class max_general_2x2(nn.Module):
	"""docstring for max_general_2x2"""
	def __init__(self,size_in,size_out):
		super(max_general_2x2, self).__init__()
		self.size_in = size_in[1:]
		self.W = self.size_in[2] # assuming width equal height
		self.H = self.size_in[1] # assuming width equal height
		self.channels = self.size_in[0]
		self.size_out = size_out[1:]
		self.n_blocks = self.channels*self.H*self.W//4

		self.B = nn.Linear(3*self.n_blocks, self.n_blocks)
		self.B.weight.data = torch.zeros_like(self.B.weight.data)
		tempB = torch.Tensor([[1,1,-1]])
		for i in range(self.n_blocks):
			self.B.weight.data[1*i:1*(i+1), 3*i:3*(i+1)] = tempB

		self.B.bias.data = 0*self.B.bias.data

		self.AD = nn.Linear(6*self.n_blocks,3*self.n_blocks)
		self.AD.weight.data = torch.zeros_like(self.AD.weight.data)
		tempAD = torch.Tensor([[ 1.,  1., -1., -1., -1.,  1.],
							[ 0.,  0.,  0.,  1.,  1., -1.],
							[ 0.,  0.,  0., -1., -1.,  1.]])
		for i in range(self.n_blocks):
			self.AD.weight.data[3*i:3*(i+1), 6*i:6*(i+1)] = tempAD

		self.AD.bias.data = 0*self.AD.bias.data		
		

		self.C = nn.Linear(4*self.n_blocks,6*self.n_blocks)
		self.C.weight.data = torch.zeros_like(self.C.weight.data)
		tempC1 = torch.Tensor([[1, -1],
							   [0, 1],
							   [0, -1],
							   [0, 0],
							   [0, 0],
							   [0, 0]])

		tempC2 = torch.Tensor([[0, 0],
							   [0, 0],
							   [0, 0],
							   [1, -1],
							   [0, 1],
							   [0, -1]])

		for i in range(self.n_blocks):
			channel_ind = i//((self.W*self.H)//4)
			channel_local_i = i%((self.W*self.H)//4)
			offset1 = 2*self.W*(channel_local_i//(self.W//2)) + channel_ind*self.W*self.H
			local_i = channel_local_i%(self.W//2)
			self.C.weight.data[6*i:6*(i+1), offset1+2*(local_i):offset1+2*(local_i+1)] = tempC1

			offset2 = self.W + offset1
			self.C.weight.data[6*i:6*(i+1), offset2 + 2*(local_i):offset2 + 2*(local_i+1)] = tempC2

		self.C.bias.data = 0*self.C.bias.data
		# Make the parameters of this layer non-learnable
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		out = nn.Sequential(*[
			Flatten(),
			self.C,
			nn.ReLU(inplace=True),
			self.AD,
			nn.ReLU(inplace=True),
			self.B
			])		
		out = out(x)
		shape = out.shape[:1] + self.size_out

		return out.view(shape)


if __name__ == "__main__":
	x=torch.Tensor([[100, 20], [4, 8]]).unsqueeze(0) 
	model = nn.Sequential(max_2x2())
	print(model(x))


	x= torch.rand([10,32,32]).unsqueeze(0)
	size_out = (x.size(0),x.size(1),x.size(2)//2,x.size(3)//2)
	model = nn.Sequential(max_general_2x2(size_in=x.size(), size_out=size_out))
	print(x)
	print(model(x))
	mp = nn.MaxPool2d(2)
	mp(x)
	embed()


