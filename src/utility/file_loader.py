#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from collections import Counter
import torch


class File_loader:
    def __init__(self):
        self.raw_path = ''
        self.stop_path = ''
        self.stopwords = []
        self.labels = []
        self.sentences = []
        self.raw_sentences = []
        self.words = []
        self.vocab = []
        self.vocab_size = 0
        # the following attributes could be deleted after the development
        self.word2idx = []
        self.label2idx = []
        self.encoded_sentences = []
        self.encoded_labels = []
        self.encoded_data = []

    def read_file(self, raw_path, stop_path=''):
        """
        read the raw questions from the question file raw_data.txt
        input empty string if the stopwords are not needed
        and split the data into label part and sentence part
        :param raw_path: The path to the raw question file
        :param stop_path: The path to the stopwords file
        """
        self.raw_path = raw_path
        self.stop_path = stop_path
        # read the stopwords if required
        if stop_path != '':
            self.read_stopwords()
        with open(self.raw_path, 'r') as f:
            for line in f:
                self.raw_sentences.append(line)
                line = line.strip('\n')  # lowercase and remove \n
                result = line.split(' ', 1)
                if stop_path != '':
                    result[1] = self.remove_stopwords(result[1])
                self.labels.append(result[0])
                self.sentences.append(result[1].lower())
        # split sentences into words
        self.words = [sen.split(" ") for sen in self.sentences]

    def read_stopwords(self):
        """
        read stopwords from the stopwords.txt
        """
        with open(self.stop_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.stopwords.append(line)

    def remove_stopwords(self, sentence):
        """
        This function removes the stopwords from the sentences
        :param sentence: a single sentence
        :return: sentence without stopwords
        """
        words = sentence.split(' ')
        new_sentence = []
        for w in words:
            if w not in self.stopwords:
                new_sentence.append(w)
        return ' '.join(new_sentence)

    def split_dataset(self, train_file_name, dev_file_name, ratio=0.1):
        """
        This function split the original dataset into train set and development set
        :param dev_file_name:  path to dev.txt
        :param train_file_name: path to train.txt
        :param ratio: The ratio of train data size and development data size
        :return:
        """
        assert len(self.raw_sentences) > 0, "Please read the raw data to split!"
        data = np.array(self.raw_sentences)
        # make random indices
        random_indices = np.random.permutation(len(data))
        dev_size = int(len(data) * ratio)
        dev_indices = random_indices[:dev_size]
        train_indices = random_indices[dev_size:]
        train_data = data[train_indices]
        dev_data = data[dev_indices]

        # write the train data into the file
        with open(train_file_name, 'w') as f:
            for i in range(len(train_data)):
                f.write(train_data[i].strip('\n') + '\n')
        # write the dev data into the file
        with open(dev_file_name, 'w') as f2:
            for j in range(len(dev_data)):
                f2.write(dev_data[j].strip('\n') + '\n')

    def read_vocab(self, path):
        """
        This function reads vocabulary file
        :param path: The path to vocabulary.txt
        :return:
        """
        with open(path, 'r') as f:
            for line in f:
                self.vocab.append(line.strip('\n'))

    def create_vocab(self, path=''):
        """
        This function creates the vocabulary for the raw_data.txt
        and store the vocabularies in the vocabulary.txt
        NOTE THAT THIS FUNCTION IS ONLY FOR THE raw_data.txt
        :param path: The path to store the vocabulary.txt
        """
        assert len(self.words) > 0, "Please read the raw data then call this function"
        words = sum(self.words, [])
        self.vocab = Counter(words)
        self.vocab = sorted(self.vocab, key=self.vocab.get, reverse=True)
        self.vocab_size = len(self.vocab)
        if path != '':
            # store vocabulary if path is not empty
            with open(path, 'w') as f:
                for i in range(self.vocab_size):
                    f.write(self.vocab[i] + '\n')

    def get_encoded_data(self, padding):
        """
        This function encodes the sentences and labels
        :return: The encoded data in the format of
        [([encoded sentence], encoded label),...] -> example. [([3,1,4,5], 2),...]
        """
        # encode sentences
        assert len(self.words) > 0, "Please read the raw data!"
        assert len(self.vocab) > 0, "Please read the vocabulary!"
        self.word2idx = {w: idx for idx, w in enumerate(self.vocab)}
        self.encoded_sentences = [[self.word2idx[w] for w in word] for word in self.words]
        # encode labels
        labels = Counter(self.labels)
        labels = sorted(labels, key=labels.get, reverse=True)
        self.label2idx = {l: idx for idx, l in enumerate(labels)}
        self.encoded_labels = [self.label2idx[label] for label in self.labels]
        # padding for the short sentences
        for en_sen in self.encoded_sentences:
            while len(en_sen) < padding:
                en_sen.append(-1)
        # put the encoded labels and sentences together
        for i in range(len(self.encoded_labels)):
            en_sen = torch.LongTensor(self.encoded_sentences[i])
            en_lab = torch.LongTensor([self.encoded_labels[i]])
            data_pair = (en_sen, en_lab)
            self.encoded_data.append(data_pair)

        return self.encoded_data

