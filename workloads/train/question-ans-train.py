"""
TODO model - Bert
"""

import os
import sys
import time
import argparse
import evaluate
import numpy as np

#import custom modules
sys.path.append('../..')
from utils.util import count_parameters
from utils.parser import MLParser
from utils.profiler import MLProfiler

import os
import sys
import time
import argparse
import evaluate
import numpy as np

#import custom modules
sys.path.append('../..')
from utils.util import count_parameters
from utils.parser import MLParser
from utils.profiler import MLProfiler

def train(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    pass

if __name__ == "__main__":
    args = MLParser(mode="train").get_args()
    profiler = MLProfiler(args) if args.profile else None
    train(args, profiler)