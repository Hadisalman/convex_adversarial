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

from copy import deepcopy
from model_factory import select_cifar_model
import problems as pblm
from trainer import *
import math
import numpy as np
import os

from maxpool_as_dense_layer import max_general_2x2
from IPython import embed

def get_linear_maxpool(size_in, size_out):
	model = max_general_2x2(size_in, size_out)

	B = model.B
	AD = model.AD
	C = model.C
	return [pblm.Flatten(), C, nn.ReLU(inplace=True), AD, nn.ReLU(inplace=True), B]

def transform_network(net, X):
	transformed_net=nn.Sequential()

	sizes=[]
	Xc = X.cuda()
	for layer in net:
		Xc = layer(Xc)
		sizes.append(Xc.shape)

	i=1		
	for layer in net:
		if isinstance(layer, nn.Linear): 
			transformed_net.add_module(str(i), layer)
		elif isinstance(layer, nn.Conv2d): 
			transformed_net.add_module(str(i), layer)
		elif isinstance(layer, nn.ReLU):   
			transformed_net.add_module(str(i),layer)	
		elif 'Flatten' in (str(layer.__class__.__name__)): 
			transformed_net.add_module(str(i),layer)
		elif isinstance(layer, nn.BatchNorm2d):
			transformed_net.add_module(str(i),layer)
		elif isinstance(layer, nn.MaxPool2d):
			mp_layers = get_linear_maxpool(sizes[i-2], sizes[i-1])
			for it,l in enumerate(mp_layers):
				transformed_net.add_module(str(i)+'_'+str(it),l)
		else:
			print(layer)
			raise ValueError("No module for layer {}".format(str(layer.__class__.__name__)))
		i+=1
	return transformed_net

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
	
	# convert the model to be compattible with the convex_adversarial package
	model_new = model.module.features
	model_new.add_module('flatten', pblm.Flatten())
	model_new.add_module('classifier', model.module.classifier)
	transformed_network = transform_network(model_new, X[:1,:])
	embed()
	err = evaluate_baseline(test_loader, transformed_network)
	epsilons = [pixel/255.0 for pixel in range(0,16,1)]
	for it, epsilon in enumerate(epsilons):
		print('epsilon = ',epsilon)
		# err_madry = evaluate_madry(test_loader, model_new, epsilon)

		err_wong = evaluate_robust(test_loader, model, epsilon, 
		   norm_type=args.norm_test, bounded_input=True, **kwargs)
		
		writer.add_scalar('verification/error', err, it)   
		writer_madry.add_scalar('verification/error', err_madry, it)   
		# writer_wong.add_scalar('verification/error', err_wong, it)   
	writer.close()
	writer_madry.close()
	writer_wong.close()