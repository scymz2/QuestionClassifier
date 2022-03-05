#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
import random
from sklearn.metrics import accuracy_score
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


# def compute_acc(model, data_loader):
#     correct = 0
#     total = 0
#     for features, labels in data_loader:
#         outputs = model(features)
#         pred = outputs.max(1, keepdim=True)[1]
#         correct += pred.eq(labels.view_as(pred)).sum().item()
#         total += features.shape[0]
#         return correct / total

def compute_acc(outputs, labels):
        pred = outputs.max(1, keepdim=True)[1]
        acc = accuracy_score(labels, pred)
        return acc


def train(config, vocab, batch_size=40, num_epoch=11, mode='develop'):
    train_data = get_encoded_data(config['PATH']['path_train'], config['PATH']['path_vocab'], config['PARAMETER']['padding'])

    dev_data = get_encoded_data(config['PATH']['path_dev'], config['PATH']['path_vocab'], config['PARAMETER']['padding'])

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


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epoch = 0
    iters = 0
    num_iters, losses, train_accs = [], [], []
    for epoch in range(num_epoch):
        for train_features, train_labels in iter(train_loader):
            # drop last batch
            if len(train_labels) != batch_size:
                continue

            # train model
            outputs = model(train_features)                         # forward pass
            loss = criterion(outputs, train_labels)                 # compute loss
            loss.backward()                                         # backward pass
            optimizer.step()                                        # update weights
            optimizer.zero_grad()                                   # clean gradients

            # record training information
            num_iters.append(epoch)                                 # counts of iterations
            losses.append(float(loss/batch_size))                   # average loss of the batch
            train_acc = compute_acc(outputs, train_labels)
            train_accs.append(train_acc)                            # training accuracy

            # print information
            if mode == 'develop':
                print('Epoch: ' + str(epoch) +
                      ' Train Set Accuracy: ' + str(train_acc)
                      )
            iters += 1
        epoch += 1

    if mode == 'develop':
        epoch = 0
        iters = 0
        val_accs = []
        for epoch in range(num_epoch):
            for val_features, val_labels in iter(val_loader):
                # drop last batch
                if len(val_labels) != batch_size:
                    continue

                outputs = model(val_features)
                val_acc = compute_acc(outputs, val_labels)
                val_accs.append(val_acc)  # validation accuracy

                # print information
                if mode == 'develop':
                    print('Epoch: ' + str(epoch) +
                          ' Validation Set Accuracy: ' + str(val_acc)
                          )

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

