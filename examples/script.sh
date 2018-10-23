python verify_network.py --data_dir data/ --output_dir verification_exp/small --model small  &&
python verify_network.py --data_dir data/ --output_dir verification_exp/large --model large &&
python verify_network.py --data_dir data/ --output_dir verification_exp/wide1 --model wide --model_factor 1 &&
python verify_network.py --data_dir data/ --output_dir verification_exp/wide2 --model wide --model_factor 2 &&
python verify_network.py --data_dir data/ --output_dir verification_exp/wide4 --model wide --model_factor 4 &&
python verify_network.py --data_dir data/ --output_dir verification_exp/wide8 --model wide --model_factor 8 &&
python verify_network.py --data_dir data/ --output_dir verification_exp/wide16 --model wide --model_factor 16 &&
python verify_network.py --data_dir data/ --output_dir verification_exp/deep1 --model deep --model_factor 1 &&
python verify_network.py --data_dir data/ --output_dir verification_exp/deep2 --model deep --model_factor 2 &&
python verify_network.py --data_dir data/ --output_dir verification_exp/deep3 --model deep --model_factor 3
