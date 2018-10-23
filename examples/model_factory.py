from cifar_models import * #import all models
import problems as pblm


def select_cifar_model(m, pretrained=False): 
	if m == 'large':
		# raise ValueError
		model = pblm.cifar_model_large().cuda()

	elif m== 'small': 
		model = pblm.cifar_model().cuda() 

##################

	elif m == 'resnet':
		model = pblm.cifar_model_resnet(N=args.resnet_N, factor=args.resnet_factor).cuda()
	
	elif m == 'ResNet18':
		print("Using ResNet18 model modified for the cifar dataset")
		model = nn.DataParallel(ResNet18().cuda())

	elif m == 'ResNet34':
		print("Using ResNet34 model modified for the cifar dataset")
		model = nn.DataParallel(ResNet34().cuda())

	elif m == 'ResNet50':
		print("Using ResNet50 model modified for the cifar dataset")
		model = nn.DataParallel(ResNet50().cuda())

	elif m == 'ResNet101':
		print("Using ResNet101 model modified for the cifar dataset")
		model = nn.DataParallel(ResNet101().cuda())

	elif m == 'ResNet152':
		print("Using ResNet152 model modified for the cifar dataset")
		model = nn.DataParallel(ResNet152().cuda())

##################

	elif m == 'vgg11':
		print("Using vgg11 model modified for the cifar dataset")
		model = nn.DataParallel(vgg11().cuda())
		if pretrained:
			print('loading a pretrained model...' )
			checkpoint = torch.load('./pretrained_models/vgg11_best.pth')
			model.load_state_dict(checkpoint['state_dict'])	
			accuracy = (1-checkpoint['err'])*100
			print('Model successfully loaded and the test accuracy' 
				  'on CIFAR-10 is {0} %'.format(accuracy) )
			
	elif m == 'vgg11_bn':
		print("Using vgg1 with batch norm modified for the cifar dataset")
		model = nn.DataParallel(vgg11_bn().cuda())

	elif m == 'vgg13':
		print("Using vgg13 model modified for the cifar dataset")
		model = nn.DataParallel(vgg13().cuda())

	elif m == 'vgg13_bn':
		print("Using vgg13 with batch norm modified for the cifar dataset")
		model = nn.DataParallel(vgg13_bn().cuda())

	elif m == 'vgg16':
		print("Using vgg16 model modified for the cifar dataset")
		model = nn.DataParallel(vgg16().cuda())

	elif m == 'vgg16_bn':
		print("Using vgg16 with batch norm modified for the cifar dataset")
		model = nn.DataParallel(vgg16_bn().cuda())

	elif m == 'vgg19':
		print("Using vgg19 model modified for the cifar dataset")
		model = nn.DataParallel(vgg19().cuda())

	elif m == 'vgg19_bn':
		print("Using vgg19 with batch norm modified for the cifar dataset")
		model = nn.DataParallel(vgg19_bn().cuda())

##################

	elif m == 'DenseNet121':
		print("Using DenseNet121 model modified for the cifar dataset")
		model = nn.DataParallel(DenseNet121().cuda())

	elif m == 'DenseNet169':
		print("Using DenseNet169 model modified for the cifar dataset")
		model = nn.DataParallel(DenseNet169().cuda())

	elif m == 'DenseNet201':
		print("Using DenseNet201 model modified for the cifar dataset")
		model = nn.DataParallel(DenseNet201().cuda())

	else:
		raise Exception('Please specify a valid model.')
	return model



