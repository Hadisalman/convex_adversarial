import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tensorboardX import SummaryWriter

import setproctitle

from model_factory import select_cifar_model
import problems as pblm
from trainer import *
import math
import numpy as np
import os

from IPython import embed

if __name__ == "__main__":
	args = pblm.argparser(verbose=100, opt='sgd', lr=0.1, epochs=50, batch_size=128)
	
	writer = SummaryWriter(os.path.join(args.output_dir, "normal"))

	print("saving file to {}".format(args.prefix))
	setproctitle.setproctitle(args.prefix)
	train_log = open(os.path.join(args.output_dir, "train.log"), "w")
	test_log = open(os.path.join(args.output_dir, "test.log"), "w")

	train_loader, test_loader = pblm.cifar_loaders(args.batch_size, data_directory=args.data_dir)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	for X,y in train_loader: 
		break
	kwargs = pblm.args2kwargs(args, X=Variable(X.cuda()))
		
	# The test_loader's batch_size varies with the model size for CUDA memory issues  
	model = select_cifar_model(args.model)
		
	best_err = 1
	best_epoch = 1
	best_model_state_dict = model.state_dict()

	if args.load:
		checkpoint = torch.load('./best.t7')
		model.load_state_dict(checkpoint['net'])
		best_epoch = checkpoint['epoch']
		best_err = evaluate_baseline(test_loader, model, best_epoch, test_log, 
							verbose=False)
	
	if args.train:
		lr_schedule = [args.lr, args.lr/10, args.lr/100]
		
		for lr in lr_schedule:	
			start_epoch = best_epoch
			
			if args.opt == 'adam':
				opt = optim.Adam(model.parameters(), lr=lr)
			elif args.opt == 'sgd': 
				opt = optim.SGD(model.parameters(), lr=lr, 
								momentum=args.momentum,
								weight_decay=args.weight_decay)
			else: 
				raise ValueError("Unknown optimizer")

			for epoch in range(start_epoch, start_epoch + args.epochs):
				train_baseline(train_loader, model, opt, epoch, train_log,
								writer, args.verbose)
				err = evaluate_baseline(test_loader, model, epoch, test_log, 
									args.verbose, writer=writer)

				if err < best_err:
					best_err = err
					best_model_state_dict = model.state_dict()
					best_epoch = epoch
					torch.save({
						'state_dict' : best_model_state_dict, 
						'err' : best_err,
						'epoch' : best_epoch,
						}, os.path.join(args.output_dir, "best.pth"))
				
				print('Best accuracy so far = ', (1-best_err.item())*100,'% \n')
				torch.save({
					'state_dict': model.state_dict(),
					'err' : err,
					'epoch' : epoch,
					}, os.path.join(args.output_dir, "checkpoint.pth"))

			# load best model
			print('Loading the best model from lr {0}'.format(lr))
			model.load_state_dict(best_model_state_dict)
			print(best_epoch)

	print("---------------------------------------------------")
	print('The best model test accuracy is = {error:.4f} %'.format(error=(1.0-best_err.item())*100))	
	print("---------------------------------------------------")

