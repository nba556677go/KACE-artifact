import sys
sys.path.append('../../..')
from utils.loaddata import train_test_split_multiinstance
import argparse

#create arg parser for all parameters
parser = argparse.ArgumentParser()
parser.add_argument("--data","-d", type=str, help="Path to the dataset")
parser.add_argument("--train_file", type=str, help="Path to the training set")
parser.add_argument("--test_file", type=str, help="Path to the testing set")
parser.add_argument("--random_seed", "-rd", type=int, help="Random seed for the split")
parser.add_argument("--train_ratio", type=float, help="Ratio of the training set")
args = parser.parse_args()
train_test_split_multiinstance(data=args.data, 
                                    train_outname=args.train_file, 
                                    test_outname=args.test_file, 
                                    random_seed=args.random_seed, train_ratio=args.train_ratio)