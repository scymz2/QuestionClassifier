#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
import random
import sklearn.metrics as metrics
import configparser
import argparse
import os

from utility.file_loader import*
from model import*
from utility.pre_train import*

torch.manual_seed(1)
random.seed(1)

def load_raw_file(config):
    '''
    create a File_loader object,
    in order to read the raw data file, create the vocabulary and label file, and split the data into train set and validation set.
    :return: the whole vocabulary (from the raw data)
    '''
    loader = File_loader()
    loader.read_file(config['PATH']['path_raw'], config['PATH']['path_stopwords'])
    loader.split_dataset(config['PATH']['path_train'], config['PATH']['path_dev'])
    loader.create_vocab_and_label(config['PATH']['path_vocab'],config['PATH']['path_label'])
    vocab = get_vocab(config)

    return vocab

def get_vocab(config):
    '''
    read the existing vocabulary file and return the vocabulary
    '''
    vocab = []
    with open(config['PATH']['path_vocab'], 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))
    return vocab

def get_encoded_data(path_file, path_vocab,path_label, path_stop, padding):
    '''
    create a File_loader object, in order to read the file (train/validation/test..)
    and encode the labels and sentences in the file.
    after encoding, add the paddings to make each sentence with the same length, padding=0.
    :return: the encoded data
    '''
    loader = File_loader()
    loader.read_file(path_file,path_stop)
    loader.read_vocab_and_label(path_vocab, path_label)
    data = loader.get_encoded_data(int(padding))

    return data

def compute_acc(outputs, target):
    '''
    compute the accuracy and the f1_score of the model using sklearn.metrics.
    '''
    pred = outputs.max(1, keepdim=True)[1]
    pred = torch.reshape(pred,(1,len(pred)))[0]
    acc = metrics.accuracy_score(target, pred)
    f1_score = metrics.f1_score(target, pred, average='weighted')
    return acc, f1_score

def train(config, vocab):
    train_data= get_encoded_data(config['PATH']['path_train'], config['PATH']['path_vocab'], config['PATH']['path_label'],  config['PATH']['path_stopwords'], config['SETTING']['padding'])

    dev_data= get_encoded_data(config['PATH']['path_dev'], config['PATH']['path_vocab'], config['PATH']['path_label'], config['PATH']['path_stopwords'],config['SETTING']['padding'])


    test_data = get_encoded_data(config['PATH']['path_test'], config['PATH']['path_vocab'],
                                 config['PATH']['path_label'], config['PATH']['path_stopwords'],
                                 config['SETTING']['padding'])


    pre_train_loader = Pre_train_loader()
    pre_train_weight = pre_train_loader.load_pretrain(config['PATH']['path_pre_train'], vocab)
    vocab_size = len(vocab) + 1

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
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses, train_accs = [], []

    for epoch in range(num_epoch):
        train_out = torch.tensor([])
        train_label = torch.tensor([])

        for train_features, train_labels in iter(train_loader):
            if len(train_labels) != batch_size:
                continue
            output = model(train_features)
            loss = criterion(output, train_labels)                    # compute loss
            loss.backward()                                         # backward pass
            optimizer.step()                                        # update weights
            optimizer.zero_grad()                                   # clean gradients
            losses.append(float(loss)/batch_size)                   # average loss of the batch

            acc, _ = compute_acc(output, train_labels)
            train_accs.append(acc)
            train_out = torch.cat((train_out, output), 0)
            train_label = torch.cat((train_label, train_labels))

        train_acc, train_f1 = compute_acc(train_out, train_label)
        print('Epoch: ', epoch, '\nTrain: Accuracy: ', train_acc, ', Train f1: ', train_f1)

        dev_out = torch.tensor([])
        dev_label = torch.tensor([])
        for dev_feats, dev_labels in iter(dev_loader):
            out = model(dev_feats)
            dev_out = torch.cat((dev_out, out), 0)
            dev_label = torch.cat((dev_label, dev_labels))

        dev_acc, dev_f1 = compute_acc(dev_out, dev_label)
        print('Validation Accuracy: ', dev_acc, ', Validation f1: ', dev_f1)

    model_path = config['PATH']['path_model']
    torch.save(model, model_path)

def test(config):
    model = torch.load(config['PATH']['path_model'])
    test_data = get_encoded_data(config['PATH']['path_test'], config['PATH']['path_vocab'],
                                  config['PATH']['path_label'], config['PATH']['path_stopwords'],
                                  config['SETTING']['padding'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=int(config['STRUCTURE']['batch_size']))

    labels = []
    with open(config['PATH']['path_label'], 'r') as f:
        for line in f:
            labels.append(line.strip('\n'))

    test_out = torch.tensor([])
    test_label = torch.tensor([])
    for test_feats, test_labels in iter(test_loader):
        out = model(test_feats)
        test_out = torch.cat((test_out, out), 0)
        test_label = torch.cat((test_label, test_labels))

    # write the output file
    test_acc, test_f1 = compute_acc(test_out, test_label)
    pred = test_out.max(1,keepdim=True)[1]
    with open(config['PATH']['path_output'], 'w') as f:
        for p in pred:
            f.write(labels[p] + '\n')
        f.write('\nAccuracy:' + str(test_acc))
        f.write('\nF1 score:' + str(test_f1))

    print('Test Accuracy: ', test_acc, ', Test f1: ', test_f1)


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

    # if the train and validation sets have not been split,
    # read the raw file, split the data and create the vocabulary and label file.
    # if the train and vali sets have been splitted, read the vocabulary file and get the vocabulary

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
