import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tensorboardX import SummaryWriter

import setproctitle

import problems as pblm
from trainer import *
import math
import numpy as np
import os
import scipy.io as sio

from IPython import embed


def select_model(args):
	m = args.model 
	if m == 'large': 
		model = pblm.mnist_model_large().cuda()
		_, test_loader = pblm.mnist_loaders(8, data_directory=args.data_dir)
	elif m == 'wide':
		print("Using wide model with model_factor={}".format(args.model_factor))
		_, test_loader = pblm.mnist_loaders(64//args.model_factor,
										data_directory=args.data_dir)
		model = pblm.mnist_model_wide(args.model_factor).cuda()
	elif m == 'deep':
		print("Using deep model with model_factor={}".format(args.model_factor))
		_, test_loader = pblm.mnist_loaders(64//(2**args.model_factor),
										data_directory=args.data_dir)
		model = pblm.mnist_model_deep(args.model_factor).cuda()
	
	elif m == 'n1':
		n1 = sio.loadmat('./MIPVerify_data/weights/mnist/n1.mat')

		b1 = n1['fc1/bias']
		b2 = n1['fc2/bias']
		b3 = n1['logits/bias']
		w1 = n1['fc1/weight']
		w2 = n1['fc2/weight']
		w3 = n1['logits/weight']	
		model = nn.Sequential(pblm.Flatten(),
							  nn.Linear(w1.shape[0],w1.shape[1]),
							  nn.ReLU(),
							  nn.Linear(w2.shape[0],w2.shape[1]),
							  nn.ReLU(),
							  nn.Linear(w3.shape[0],w3.shape[1]),
							  )
		model[1].weight.data = torch.Tensor(w1).t()
		model[1].bias.data = torch.Tensor(b1)
		model[3].weight.data = torch.Tensor(w2).t()
		model[3].bias.data = torch.Tensor(b2)
		model[5].weight.data = torch.Tensor(w3).t()
		model[5].bias.data = torch.Tensor(b3)
		model = model.cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)

		# embed()

	elif m == 'n2':
		pass

	elif m == 'aditi':
		"""
		Converts weights provided by authors in `B1.npy`, `B2.npy`, `W1.npy` 
		and `W2.npy` into a single `two-layer.mat` file.
		"""
		b1 = np.load("./MIPVerify_data/weights/mnist/RSL18a/linf0.1/B1.npy")
		b2 = np.load("./MIPVerify_data/weights/mnist/RSL18a/linf0.1/B2.npy")
		w1 = np.load("./MIPVerify_data/weights/mnist/RSL18a/linf0.1/W1.npy")
		w2 = np.load("./MIPVerify_data/weights/mnist/RSL18a/linf0.1/W2.npy")

		model = nn.Sequential(pblm.Flatten(),
							  nn.Linear(w1.shape[0],w1.shape[1]),
							  nn.ReLU(),
							  nn.Linear(w2.shape[0],w2.shape[1])
							  )
		model[1].weight.data = torch.Tensor(w1).t()
		model[1].bias.data = torch.Tensor(b1)
		model[3].weight.data = torch.Tensor(w2).t()
		model[3].bias.data = torch.Tensor(b2)
		model = model.cuda()
		# embed()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)

	elif m == 'small':
		model = pblm.mnist_model().cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)
		# path = '../models_scaled/mnist_small_0_1.pth'
		path = './weights/cnn_small_madry.pth'
	
	elif m == 'layers2_nodes10':
		model = pblm.layers2_nodes10(784).cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)

	elif m == 'layers2_nodes100':
		model = pblm.layers2_nodes100(784).cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)
		path = './weights/layers2_nodes100_madry_0.05.pth'
		# path = './weights/layers2_nodes100_normal.pth'

	elif m == 'layers10_nodes100':
		model = pblm.layers10_nodes100(784).cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)

	elif m == 'layers10_nodes500':
		model = pblm.layers10_nodes500(784).cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)

	elif m == 'layers10_nodes1000':
		model = pblm.layers10_nodes1000(784).cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)

	elif m == 'layers10_nodes5000':
		model = pblm.layers10_nodes5000(784).cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)		

	if args.load:
		# checkpoint = torch.load('./master_seed_1_epochs_100.pth')
		checkpoint = torch.load(path)['state_dict'][0]
		model.load_state_dict(checkpoint)
		# best_epoch = checkpoint['epoch']

	return model, test_loader


if __name__ == "__main__":
	args = pblm.argparser(opt='adam', verbose=200)
	
	train_loader, _ = pblm.mnist_loaders(args.batch_size, data_directory=args.data_dir)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	for X,y in train_loader: 
		break
	kwargs = pblm.args2kwargs(args, X=Variable(X.cuda()))
		
	# The test_loader's batch_size varies with the model size for CUDA memory issues  
	model, test_loader = select_model(args)
	
	normal_error = evaluate_baseline(test_loader, model, verbose=False)
	print('normal error: ', normal_error)
	t = 0

	# epsilons = np.linspace(1e-1, 1e-3, num=20)
	epsilon = args.epsilon
	niters=10 
	alpha=0.1

	print('Validating for epsilon= ', epsilon)
	adv_examples_indices = []
	err_pgd = 0
	for X,y in test_loader:
		X,y = X.cuda(), y.cuda()
		out = model(X)
		ce = nn.CrossEntropyLoss()(out, y)
		# err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

		X_pgd = Variable(X.data, requires_grad=True)
		for i in range(niters): 
			opt = optim.Adam([X_pgd], lr=1e-3)
			opt.zero_grad()
			loss = nn.CrossEntropyLoss()(model(X_pgd), y)
			loss.backward()
			eta = alpha*X_pgd.grad.data.sign()
			X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
			
			# adjust to be within [-epsilon, epsilon]
			eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
			
			# ensure valid pixel range
			X_pgd = Variable(torch.clamp(X.data + eta, 0, 1.0), requires_grad=True)

		mismatch = model(X_pgd).data.max(1)[1] != y.data
		adv_examples_indices += mismatch.cpu().numpy().tolist()
		err_pgd += (mismatch).float().sum()


	embed()
	err_pgd /= 10000
	print("PGD error is {}%".format(err_pgd*100))
	np.savetxt('adver_examples_indices.txt',adv_examples_indices)
	print('Saved the indeices of the adversarial exmaples to "{}"'.format('adver_examples_indices.txt'))
	# err_madry = evaluate_madry(test_loader, model, epsilon, 
	# 					 t, test_log, args.verbose)

	# err_wong = evaluate_robust(test_loader, model, epsilon,
	#    t, test_log, args.verbose, args.real_time,
	#    norm_type=args.norm_test, bounded_input=True, **kwargs)
