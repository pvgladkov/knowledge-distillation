# coding: utf-8

from __future__ import unicode_literals

import os

TRAIN_TEST_RATIO = 0.2


ROOT_DATA_PATH = '/data/sst2'
TRAIN_FILE = os.path.join(ROOT_DATA_PATH, 'train.tsv')
TEST_FILE = os.path.join(ROOT_DATA_PATH, 'test.tsv')
DEV_FILE = os.path.join(ROOT_DATA_PATH, 'dev.tsv')


bert_settings = {
    'max_seq_length': 128,
    'num_train_epochs': 4,
    'train_batch_size': 16,
    'eval_batch_size': 16,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'test_size': TRAIN_TEST_RATIO
}

lstm_settings = {
    'max_seq_length': 128,
    'num_train_epochs': 10,
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'test_size': TRAIN_TEST_RATIO
}

distillation_settings = {
    'max_seq_length': 128,
    'num_train_epochs': 20,
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'test_size': TRAIN_TEST_RATIO
}

