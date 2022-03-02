#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import random
import numpy as np
import torch.nn as nn
import utilty.preprocessing
from collections import Counter
torch.manual_seed(1)
random.seed(1)

def train():
    pass

def test():
    pass

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Parse command line arguments.')
    parser.add_argument('--train', action='')
    parser.add_argument('--test', action='')
    parser.add_argument('--config', help='Configure parameters.')
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()

    # load configurations
    configs = load_configs()

    # preprocess


    # train
    if args.train:
        train()

    # test
    if args.test:
        test()


