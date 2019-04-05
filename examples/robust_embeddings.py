# import waitGPU
# import setGPU
# waitGPU.wait(utilization=50, available_memory=10000, interval=60)
# waitGPU.wait(gpu_ids=[1,3], utilization=20, available_memory=10000, interval=60)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

# from tensorboardX import SummaryWriter

import setproctitle

import problems as pblm
from trainer import *
import math
import numpy as np
import os

from IPython import embed

def model():
    # model1 = nn.Sequential(
    # nn.Conv2d(1, 16, 4, stride=2, padding=1),
    # nn.ReLU(),
    # nn.Conv2d(16, 32, 4, stride=2, padding=1),
    # nn.ReLU(),
    # nn.Conv2d(32, 64, 1, stride=2, padding=1),
    # )
    # model2 = nn.Sequential(
    # nn.ReLU(),
    # pblm.Flatten(),
    # nn.Linear(1600,100),
    # nn.ReLU(),
    # nn.Linear(100, 10)
    # )

    model1 = nn.Sequential(
    nn.Conv2d(1, 16, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 4, stride=2, padding=1),
    nn.ReLU(),
    pblm.Flatten(),
    nn.Linear(1568, 100),
    )
    model2 = nn.Sequential(
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
    )

    output_model = nn.Sequential(model1, model2)
    return output_model.cuda()


if __name__ == "__main__": 
    args = pblm.argparser(opt='adam', verbose=200, starting_epsilon=0.01)
    
    # writer = SummaryWriter(os.path.join(args.output_dir, args.prefix))

    print("saving file to {}".format(args.prefix))
    setproctitle.setproctitle(args.prefix)
    train_log = open(os.path.join(args.output_dir, args.prefix + "_train.log"), "w")
    test_log = open(os.path.join(args.output_dir, args.prefix + "_test.log"), "w")

    train_loader, test_loader = pblm.mnist_loaders(args.batch_size, data_directory=args.data_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for X,y in train_loader: 
        break
    kwargs = pblm.args2kwargs(args, X=Variable(X.cuda()))
    best_err = 1

    sampler_indices = []
    net = model()

    if args.opt == 'adam': 
        opt = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        opt = optim.SGD(net.parameters(), lr=args.lr, 
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer")
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_schedule = np.linspace(args.starting_epsilon, 
                               args.epsilon, 
                               args.schedule_length)

    for t in range(args.epochs):
        lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
        if t < len(eps_schedule) and args.starting_epsilon is not None: 
            epsilon = float(eps_schedule[t])
        else:
            epsilon = args.epsilon

        # standard training
        if args.method == 'baseline': 
            train_baseline(train_loader, net, opt, t, train_log,
                            verbose=args.verbose)
            err = evaluate_baseline(test_loader, net, t, test_log,
                            verbose=args.verbose)

        # madry training
        elif args.method=='madry':
            train_madry(train_loader, net, args.epsilon, 
                        opt, t, train_log, args.verbose)
            err = evaluate_madry(test_loader, net, args.epsilon, 
                                 t, test_log, args.verbose)

        elif args.method=='composite':
            train_composite(train_loader, net, args.epsilon, 
                        opt, t, train_log, args.verbose, 
                        norm_type=args.norm_test, bounded_input=True)
            err = evaluate_madry(test_loader, net, args.epsilon, 
                                 t, test_log, args.verbose)

        # robust cascade training
        elif args.cascade > 1: 
            train_robust(train_loader, net, opt, epsilon, t,
                            train_log, args.verbose, args.real_time,
                            norm_type=args.norm_train, bounded_input=True,
                            **kwargs)
            err = evaluate_robust_cascade(test_loader, net,
               args.epsilon, t, test_log, args.verbose,
               norm_type=args.norm_test, bounded_input=True,  **kwargs)

        # robust training
        else:
            train_robust(train_loader, net, opt, epsilon, t,
               train_log, verbose=args.verbose, real_time=args.real_time,
               norm_type=args.norm_train, bounded_input=True, **kwargs)
            err = evaluate_robust(test_loader, net, args.epsilon,
               t, test_log, args.verbose, args.real_time,
               norm_type=args.norm_test, bounded_input=True, **kwargs)
        
        if err < best_err: 
            while True:
                try:
                    best_err = err
                    torch.save({
                        'state_dict' : net.state_dict(), 
                        'err' : best_err,
                        'epoch' : t,
                        'sampler_indices' : sampler_indices
                        }, os.path.join(args.output_dir, args.prefix + "_best.pth"))
                    break
                except Exception as e:
                    print(e)
        while True:
            try:
                torch.save({ 
                    'state_dict': net.state_dict(),
                    'err' : err,
                    'epoch' : t,
                    'sampler_indices' : sampler_indices
                    }, os.path.join(args.output_dir, args.prefix + "_checkpoint.pth"))
                break
            except Exception as e:
                print(e)
