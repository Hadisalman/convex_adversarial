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
	args = pblm.argparser()

	writer = SummaryWriter(os.path.join(args.output_dir, "normal"))
	writer_madry = SummaryWriter(os.path.join(args.output_dir, "madry"))
	writer_wong = SummaryWriter(os.path.join(args.output_dir, "wong"))

	train_loader, test_loader = pblm.cifar_loaders(args.batch_size, data_directory=args.data_dir)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	for X,y in train_loader: 
		break
	kwargs = pblm.args2kwargs(args, X=Variable(X.cuda()))
		
	model = select_cifar_model(args.model, pretrained=True)

	err = evaluate_baseline(test_loader, model)

	epsilons = [pixel/255.0 for pixel in range(0,9)]
	for it, epsilon in enumerate(epsilons):
		print('epsilon = ',epsilon)
		err_madry = evaluate_madry(test_loader, model, epsilon)

		# err_wong = evaluate_robust(test_loader, model.module, epsilon,
		#    best_epoch, test_log, args.verbose, args.real_time,
		#    l1_type=args.l1_test, bounded_input=True, **kwargs)
		
		writer.add_scalar('verification/error', err, it)   
		writer_madry.add_scalar('verification/error', err_madry, it)   
		# writer_wong.add_scalar('verification/error', err_wong, it)   
	writer.close()
	writer_madry.close()
	writer_wong.close()