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
    loader.create_vocab(config['PATH']['path_vocab'])
    vocab = loader.vocab

    return vocab

def get_encoded_data(path_file, path_vocab,padding):
    loader = File_loader()
    loader.read_file(path_file,'')
    loader.read_vocab(path_vocab)
    data = loader.get_encoded_data(int(padding))

    return data

def train(config, vocab):
    train_data = get_encoded_data(config['PATH']['path_train'], config['PATH']['path_vocab'], config['PARAMETER']['padding'])

    dev_data = get_encoded_data(config['PATH']['path_dev'], config['PATH']['path_vocab'], config['PARAMETER']['padding'])

    pre_train_loader = Pre_train_loader()
    pre_train_weight = pre_train_loader.get_weight(config['PATH']['path_pre_train'], vocab)
    vocab_size = len(vocab)
    print(vocab_size)


    model = Model(model=config['SETTING']['model'],
                  pre_train_weight=pre_train_weight,
                  pre_train=(config['SETTING']['pre_train']==True),
                  freeze=(config['SETTING']['freeze']==True),
                  embedding_dim=int(config['STRUCTURE']['embedding_dim']),
                  vocab_size=vocab_size,
                  hidden_dim_bilstm=config['STRUCTURE']['hidden_dim_bilstm'],
                  n_input=int(config['STRUCTURE']['n_input']),
                  n_hidden=int(config['STRUCTURE']['n_hidden']),
                  n_output=int(config['STRUCTURE']['n_output']))

def test(config):
    pass


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
    vocab = load_raw_file(config)

    # train
    if args.train:
        train(config, vocab)

    # test
    elif args.test:
        test(config)

