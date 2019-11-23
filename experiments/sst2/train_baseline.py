# coding: utf-8

from __future__ import unicode_literals, print_function
import pandas as pd
from knowledge_distillation.distilling_lstm import LSTMBaseline
from knowledge_distillation.utils import get_logger


settings = {
    'max_seq_length': 128,
    'num_train_epochs': 10,
    'train_batch_size': 32,
    'eval_batch_size': 32,
}


if __name__ == '__main__':

    logger = get_logger()

    train_df = pd.read_csv('./.data/sst/train.csv', encoding='utf-8')
    X_train = train_df['text']
    y_train = train_df['label']

    trainer = LSTMBaseline(settings, logger)
    model, vocab = trainer.train(X_train, y_train, y_train, '/app/.data/sst')

    logger.info('validate')

    test_df = pd.read_csv('./.data/sst/test.csv', encoding='utf-8')
    X_test = test_df['text']
    y_test = test_df['label']

    trainer.validate(X_test, y_test, y_test, model, vocab)

