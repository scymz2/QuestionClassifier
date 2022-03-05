#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
import random
import sklearn.metrics as metrics
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
    loader.read_file(config['PATH']['path_raw'], config['PATH']['path_stopwords'])
    loader.split_dataset(config['PATH']['path_train'], config['PATH']['path_dev'])
    loader.create_vocab_and_label(config['PATH']['path_vocab'],config['PATH']['path_label'])
    vocab = loader.vocab

    return vocab

def get_encoded_data(path_file, path_vocab,path_label, path_stop, padding):
    loader = File_loader()
    loader.read_file(path_file,path_stop)
    loader.read_vocab_and_label(path_vocab, path_label)
    data = loader.get_encoded_data(int(padding))

    return data


def compute_acc(outputs, target):
    # print(outputs)
    pred = outputs.max(1, keepdim=True)[1]
    pred = torch.reshape(pred,(1,len(pred)))[0]
    # print(target)
    # print(pred)
    acc = metrics.accuracy_score(target, pred)
    f1_score = metrics.f1_score(target, pred, average=None)
    return acc, f1_score

def train(config, vocab):
    train_data= get_encoded_data(config['PATH']['path_train'], config['PATH']['path_vocab'], config['PATH']['path_label'],  config['PATH']['path_stopwords'], config['SETTING']['padding'])

    # dev_data= get_encoded_data(config['PATH']['path_dev'], config['PATH']['path_vocab'], config['PATH']['path_label'], config['PATH']['path_stopwords'],config['SETTING']['padding'])

    pre_train_loader = Pre_train_loader()
    pre_train_weight = pre_train_loader.get_weight(config['PATH']['path_pre_train'], vocab)
    vocab_size = len(vocab)

    model = Model(model=config['SETTING']['model'],
                  pre_train_weight=pre_train_weight,
                  pre_train=(config['SETTING']['pre_train']==True),
                  freeze=(config['SETTING']['freeze']==True),
                  embedding_dim=int(config['STRUCTURE']['embedding_dim']),
                  vocab_size=vocab_size,
                  hidden_dim_bilstm=config['STRUCTURE']['hidden_dim_bilstm'],
                  n_input=int(config['STRUCTURE']['n_input']),
                  n_hidden=int(config['STRUCTURE']['n_hidden']),
                  n_output=int(config['STRUCTURE']['n_output'])
                  )

    batch_size = int(config['STRUCTURE']['batch_size'])
    num_epoch = int(config['SETTING']['epoch'])
    lr = float(config['PARAMETER']['lr'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    losses, train_accs = [], []
    for epoch in range(num_epoch):
        for train_features, train_labels in iter(train_loader):
            output = model(train_features)
            # print(output)
            loss = criterion(output, train_labels)                    # compute loss
            loss.backward()                                         # backward pass
            optimizer.step()                                        # update weights
            optimizer.zero_grad()                                   # clean gradients

            # record training information
            losses.append(float(loss/batch_size))                   # average loss of the batch
            train_acc = compute_acc(output, train_labels)
            train_accs.append(train_acc)                            # training accuracy
            train_acc, train_f1 = compute_acc(output, train_labels)

        # model_path = str('../data/model.' + config['PATH'][config['SETTING']['model']])
        # torch.save(model, model_path)
        # dev_out = model(dev_feat)
        # dev_acc, dev_f1 = compute_acc(dev_out, dev_label)

        # print information
        # print('Epoch: ' + epoch + '\nTrain: Accuracy: ' + train_acc + ', F1_score: ', + train_f1, + '\nValidation: Accuracy: '+ dev_acc, +', F1_score' + dev_f1)
        print('Epoch: ', epoch, 'Train: Accuracy: ', train_acc, ', F1_score: ', train_f1)

def test(config):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default='../src/config.ini', help='Configuration file')
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

