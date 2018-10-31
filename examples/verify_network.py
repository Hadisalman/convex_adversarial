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
		embed()
		pass

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
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)

	else:
		model = pblm.mnist_model().cuda()
		_,test_loader = pblm.mnist_loaders(args.batch_size, 
										data_directory=args.data_dir)
		

	return model, test_loader


if __name__ == "__main__": 
	args = pblm.argparser(opt='adam', verbose=200)
	
	writer = SummaryWriter(os.path.join(args.output_dir, "normal"))
	writer_madry = SummaryWriter(os.path.join(args.output_dir, "madry"))
	writer_wong = SummaryWriter(os.path.join(args.output_dir, "wong"))

	print("saving file to {}".format(args.prefix))
	setproctitle.setproctitle(args.prefix)
	train_log = open(os.path.join(args.output_dir, args.prefix + "_train.log"), "w")
	test_log = open(os.path.join(args.output_dir, args.prefix + "_test.log"), "w")

	train_loader, _ = pblm.mnist_loaders(args.batch_size, data_directory=args.data_dir)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	for X,y in train_loader: 
		break
	kwargs = pblm.args2kwargs(args, X=Variable(X.cuda()))
	
	best_err = 1
	
	# The test_loader's batch_size varies with the model size for CUDA memory issues  
	model, test_loader = select_model(args)
		
	if args.load:
		checkpoint = torch.load('./master_seed_1_epochs_100.pth')
		model.load_state_dict(checkpoint)
		# best_epoch = checkpoint['epoch']
	best_err = evaluate_baseline(test_loader, model, verbose=False)
	print('best error: ', best_err)
	embed()

	if args.train:
		if args.opt == 'adam': 
			opt = optim.Adam(model.parameters(), lr=args.lr)
		elif args.opt == 'sgd': 
			opt = optim.SGD(model.parameters(), lr=args.lr, 
							momentum=args.momentum,
							weight_decay=args.weight_decay)
		else: 
			raise ValueError("Unknown optimizer")

		for t in range(args.epochs):
			train_baseline(train_loader, model, opt, t, train_log,
							writer, args.verbose)
			err = evaluate_baseline(test_loader, model, t, test_log, 
								args.verbose)

			if err < best_err:
				best_err = err
				best_model_state_dict = model.state_dict()
				torch.save({
					'state_dict' : model.state_dict(), 
					'err' : best_err,
					'epoch' : t,
					}, os.path.join(args.output_dir, args.prefix + "_best.pth"))
				
			torch.save({ 
				'state_dict': model.state_dict(),
				'err' : err,
				'epoch' : t,
				}, os.path.join(args.output_dir, args.prefix + "_checkpoint.pth"))

		# load best model
		model.load_state_dict(best_model_state_dict)

	embed()
	epsilons = np.linspace(1e-1, 1e-3, num=20)
	for it, epsilon in enumerate(epsilons):
		err_madry = evaluate_madry(test_loader, model, epsilon, 
							 t, test_log, args.verbose)

		err_wong = evaluate_robust(test_loader, model, epsilon,
		   t, test_log, args.verbose, args.real_time,
		   norm_type=args.norm_test, bounded_input=True, **kwargs)
		
		writer.add_scalar('verification/error', err, it)   
		writer_madry.add_scalar('verification/error', err_madry, it)   
		writer_wong.add_scalar('verification/error', err_wong, it)   

	writer.close()
	writer_madry.close()
	writer_wong.close()