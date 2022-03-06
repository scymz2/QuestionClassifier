#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
import random
import sklearn.metrics as metrics
import numpy as np
import configparser
import argparse
import os

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

def get_vocab(config):
    vocab = []
    with open(config['PATH']['path_vocab'], 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))
    return vocab

def get_encoded_data(path_file, path_vocab,path_label, path_stop, padding):
    loader = File_loader()
    loader.read_file(path_file,path_stop)
    loader.read_vocab_and_label(path_vocab, path_label)
    data = loader.get_encoded_data(int(padding))

    return data


def compute_acc(outputs, target):
    pred = outputs.max(1, keepdim=True)[1]
    pred = torch.reshape(pred,(1,len(pred)))[0]
    acc = metrics.accuracy_score(target, pred)
    f1_score = metrics.f1_score(target, pred, average=None)
    return acc, f1_score

def train(config, vocab):
    train_data= get_encoded_data(config['PATH']['path_train'], config['PATH']['path_vocab'], config['PATH']['path_label'],  config['PATH']['path_stopwords'], config['SETTING']['padding'])

    dev_data= get_encoded_data(config['PATH']['path_dev'], config['PATH']['path_vocab'], config['PATH']['path_label'], config['PATH']['path_stopwords'],config['SETTING']['padding'])


    test_data = get_encoded_data(config['PATH']['path_test'], config['PATH']['path_vocab'],
                                 config['PATH']['path_label'], config['PATH']['path_stopwords'],
                                 config['SETTING']['padding'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))


    pre_train_loader = Pre_train_loader()
    pre_train_weight = pre_train_loader.load_pretrain(config['PATH']['path_pre_train'], vocab)
    vocab_size = len(vocab)

    model = Model(model=config['SETTING']['model'],
                  pre_train_weight=pre_train_weight,
                  pre_train=(config['SETTING']['pre_train']=='True'),
                  freeze=(config['SETTING']['freeze']=='True'),
                  embedding_dim=int(config['STRUCTURE']['embedding_dim']),
                  vocab_size=vocab_size,
                  hidden_dim_bilstm=int(config['STRUCTURE']['hidden_dim_bilstm']),
                  n_input=int(config['STRUCTURE']['n_input']),
                  n_hidden=int(config['STRUCTURE']['n_hidden']),
                  n_output=int(config['STRUCTURE']['n_output'])
                  )

    batch_size = int(config['STRUCTURE']['batch_size'])
    num_epoch = int(config['SETTING']['epoch'])
    lr = float(config['PARAMETER']['lr'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=len(dev_data))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    losses, train_accs = [], []

    for epoch in range(num_epoch):
        train_acc_total = 0
        itr = 0
        for train_features, train_labels in iter(train_loader):
            output = model(train_features)
            loss = criterion(output, train_labels)                    # compute loss
            loss.backward()                                         # backward pass
            optimizer.step()                                        # update weights
            optimizer.zero_grad()                                   # clean gradients

            # record training information
            losses.append(float(loss)/batch_size)                   # average loss of the batch
            acc, train_f1 = compute_acc(output, train_labels)
            train_accs.append(acc)
            train_acc_total += acc
            # print('Epoch: ', epoch, 'Train: Accuracy: ', train_acc, ', F1_score: ', train_f1)
            itr += 1
            print('Epoch: ', epoch, 'Train: Accuracy: ', acc)
        train_acc = train_acc_total/itr

        for dev_feats, dev_labels in iter(dev_loader):
            out = model(dev_feats)
            dev_acc,dev_f1 = compute_acc(out, dev_labels)
        # dev_acc_tatal = 0
        # itr = 0
        # for dev_feats, dev_labels in iter(dev_loader):
        #     out = model(dev_feats)
        #     acc,_ = compute_acc(out, dev_labels)
        #     dev_acc_tatal += acc
        #     itr+=1
        # dev_acc = dev_acc_tatal/itr
        print('Epoch: ', epoch, 'Train: Accuracy: ', train_acc, 'Validation Accuracy: ', dev_acc)


        for test_feats, test_labels in iter(test_loader):
            out = model(test_feats)
            test_acc, test_f1 = compute_acc(out, test_labels)

        print('Test Accuracy: ', test_acc)

    model_path = config['PATH']['path_model']
    torch.save(model, model_path)


def test(config):
    model = torch.load(config['PATH']['path_model'])
    test_data = get_encoded_data(config['PATH']['path_test'], config['PATH']['path_vocab'],
                                  config['PATH']['path_label'], config['PATH']['path_stopwords'],
                                  config['SETTING']['padding'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    for test_feats, test_labels in iter(test_loader):
        out = model(test_feats)
        test_acc, test_f1 = compute_acc(out, test_labels)

    print('Test Accuracy: ', test_acc)


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

    if os.path.isfile(config['PATH']['path_train']) == False:
        #load and preprocess the raw data
        vocab = load_raw_file(config)
    else:
        vocab = get_vocab(config)


    # train
    if args.train:
        train(config, vocab)

    # test
    elif args.test:
        test(config)

