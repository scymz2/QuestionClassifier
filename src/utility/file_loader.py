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
        self.all_labels = []
        self.num_of_labels = 0
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

    def read_file(self, path, stop_path=''):
        """
        read the original questions from the path file,
        and remove the stopwords from the stopwords file.
        input empty string if the stopwords are not needed
        and split the data into label part and sentence part, store them in the object.
        :param path: The path question file (raw /train /dev /test...)
        :param stop_path: The path to the stopwords file
        """
        self.raw_path = path
        self.stop_path = stop_path
        # read the stopwords if required
        if stop_path != '':
            self.read_stopwords()
        with open(self.raw_path, 'r') as f:
            for line in f:
                self.raw_sentences.append(line)
                line = line.lower().strip('\n')  # lowercase and remove \n
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

    def read_vocab_and_label(self, vocab_path, label_path):
        """
        This function reads vocabulary file/labels file
        :param label_path: The path to the labels.txt
        :param vocab_path: The path to the vocabulary.txt
        :return:
        """
        with open(vocab_path, 'r') as f:
            for line in f:
                self.vocab.append(line.strip('\n'))

        with open(label_path, 'r') as f:
            for line in f:
                self.all_labels.append(line.strip('\n'))
        self.num_of_labels = len(self.all_labels)

    def create_vocab_and_label(self, vocab_path='', label_path=''):
        """
        This function creates the vocabulary/labels for the raw_data.txt
        and store the vocabularies in the vocabulary.txt/ labels.txt
        NOTE THAT THIS FUNCTION IS ONLY FOR THE raw_data.txt
        :param vocab_path: The path to store the vocabulary.txt
        :param label_path: The path to store the labels.txt
        """
        assert len(self.words) > 0, "Please read the raw data or vocabulary file"
        assert len(self.labels) > 0, "Please read the raw data or labels file"

        words = sum(self.words, [])
        self.vocab = Counter(words)
        self.vocab = sorted(self.vocab, key=self.vocab.get, reverse=True)
        self.vocab.append('')
        self.vocab_size = len(self.vocab)
        if vocab_path != '':
            # store vocabulary if path is not empty
            with open(vocab_path, 'w') as f:
                for i in range(self.vocab_size):
                    f.write(self.vocab[i] + '\n')
                f.write('#unk#')

        self.all_labels = Counter(self.labels)
        self.all_labels = sorted(self.all_labels, key=self.all_labels.get, reverse=True)
        self.num_of_labels = len(self.all_labels)
        if label_path != '':
            # store vocabulary if path is not empty
            with open(label_path, 'w') as f:
                for i in range(self.num_of_labels):
                    f.write(self.all_labels[i] + '\n')

    def get_encoded_data(self, padding):
        """
        This function encodes the sentences and labels
        :return: The encoded data in the format of
        [([encoded sentence], encoded label),...] -> example. [([3,1,4,5], 2),...]
        """
        # encode sentences
        assert len(self.words) > 0, "Please read the raw data!"
        assert len(self.vocab) > 0, "Please read the vocabulary!"
        self.word2idx = {w: idx+1 for idx, w in enumerate(self.vocab)}
        # If it is test file, mark the unknown words with #unk# rather than index
        # if test:
        for words in self.words:
            en_sen = []
            for word in words:
                if word not in self.vocab:
                    en_sen.append(self.word2idx['#unk#'])
                else:
                    en_sen.append(self.word2idx[word])
            self.encoded_sentences.append(en_sen)
        # else:
        #     self.encoded_sentences = [[self.word2idx[w] for w in word] for word in self.words]

        # encode labels
        self.label2idx = {l: idx for idx, l in enumerate(self.all_labels)}
        self.encoded_labels = [self.label2idx[label] for label in self.labels]
        # padding for the short sentences
        for en_sen in self.encoded_sentences:
            while len(en_sen) > padding:
                en_sen.pop()
            while len(en_sen) < padding:
                en_sen.append(0)
        # put the encoded labels and sentences together
        for i in range(len(self.encoded_labels)):
            en_sen = torch.LongTensor(self.encoded_sentences[i])
            en_lab = self.encoded_labels[i]
            data_pair = (en_sen, en_lab)
            self.encoded_data.append(data_pair)

        return self.encoded_data
