#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import random
import numpy as np
import configparser
import argparse

from utility import file_loader
from collections import Counter

torch.manual_seed(1)
random.seed(1)

def train(config):
    print("train_path:", config['PATH']['path_train'])

def test(config):
    print("test_path", config['PATH']['path_train'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='../src/config.ini', help='Configuration file')
    parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
    args = parser.parse_args()

    config_path = args.config
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_path)

    # train
    if args.train:
        train(config)

    # test
    elif args.test:
        test(config)



