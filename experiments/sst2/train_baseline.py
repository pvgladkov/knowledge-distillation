# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd

from experiments.sst2.lstm_trainer import LSTMBaseline
from experiments.sst2.settings import lstm_settings, ROOT_DATA_PATH, TEST_FILE, TRAIN_FILE
from knowledge_distillation.utils import get_logger

if __name__ == '__main__':

    logger = get_logger()

    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8')
    X_train = train_df['text']
    y_train = train_df['label']

    trainer = LSTMBaseline(lstm_settings, logger)
    model, vocab = trainer.train(X_train, y_train, y_train, ROOT_DATA_PATH)

    logger.info('validate')

    test_df = pd.read_csv(TEST_FILE, encoding='utf-8')
    X_test = test_df['text']
    y_test = test_df['label']

    trainer.validate(X_test, y_test, y_test, model, vocab)

