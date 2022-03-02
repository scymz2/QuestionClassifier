#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np


class File_loader:
    def __init__(self):
        self.raw_path = ''
        self.stop_path = ''
        self.stopwords = []
        self.labels = []
        self.sentences = []
        self.raw_sentences = []

    def read_file(self, raw_path, stop_path):
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
                f.write(train_data[i])
        # write the dev data into the file
        with open(dev_file_name, 'w') as f2:
            for j in range(len(dev_data)):
                f2.write(dev_data[j])
