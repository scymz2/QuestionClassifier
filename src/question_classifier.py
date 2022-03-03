#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import random
import numpy as np
import configparser
import argparse

from utility.file_loader import*
from model import*
from utility.pre_train import*

torch.manual_seed(1)
random.seed(1)

def load_raw_file(config):
    loader = File_loader()
    loader.read_file(config['PATH']['path_raw'], '')
    loader.split_dataset(config['PATH']['path_train'], config['PATH']['path_dev'])
    print(loader.sentences)

def train(config):
    train = File_loader()
    train.read_file(config['PATH']['path_train'],'')
    data_train = train.sentences
    label_train = train.labels

    dev = File_loader()
    dev.read_file(config['PATH']['path_dev'],'')
    data_dev = dev.sentences
    label_dev = dev.labels

    pre_train_weight = Pre_train()

    model = Model(pre_train_weight, )


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

    #load and preprocess the raw data
    load_raw_file(config)

    # train
    if args.train:
        train(config)

    # test
    elif args.test:
        test(config)


