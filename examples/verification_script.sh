python verify_network_cifar.py --model vgg11 --output_dir ./verification_exp/large_networks_all/vgg11 &
python verify_network_cifar.py --model vgg11_bn --output_dir ./verification_exp/large_networks_all/vgg11_bn &&
python verify_network_cifar.py --model vgg13 --output_dir ./verification_exp/large_networks_all/vgg13 &
python verify_network_cifar.py --model vgg13_bn --output_dir ./verification_exp/large_networks_all/vgg13_bn &&
python verify_network_cifar.py --model vgg16 --output_dir ./verification_exp/large_networks_all/vgg16 &
python verify_network_cifar.py --model vgg16_bn --output_dir ./verification_exp/large_networks_all/vgg16_bn &&
python verify_network_cifar.py --model vgg19 --output_dir ./verification_exp/large_networks_all/vgg19 &
python verify_network_cifar.py --model vgg19_bn --output_dir ./verification_exp/large_networks_all/vgg19_bn &&
python verify_network_cifar.py --model ResNet18 --output_dir ./verification_exp/large_networks_all/ResNet18 &&
python verify_network_cifar.py --model ResNet34 --output_dir ./verification_exp/large_networks_all/ResNet34 &&
python verify_network_cifar.py --model ResNet50 --output_dir ./verification_exp/large_networks_all/ResNet50
