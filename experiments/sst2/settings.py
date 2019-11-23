# coding: utf-8

from __future__ import unicode_literals

import os

distillation_settings = {
    'max_seq_length': 128,
    'num_train_epochs': 10,
    'train_batch_size': 32,
    'eval_batch_size': 32,
}

bert_settings = {
    'max_seq_length': 512,
    'num_train_epochs': 3,
    'train_batch_size': 16,
    'eval_batch_size': 16,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
}
lstm_settings = {
    'max_seq_length': 128,
    'num_train_epochs': 10,
    'train_batch_size': 32,
    'eval_batch_size': 32,
}


ROOT_DATA_PATH = '/data/sst2'
TRAIN_FILE = os.path.join(ROOT_DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(ROOT_DATA_PATH, 'test.csv')
