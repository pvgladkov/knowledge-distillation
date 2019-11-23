# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd
from transformers import BertTokenizer

from experiments.sst2.bert_trainer import BertTrainer
from experiments.sst2.settings import bert_settings, ROOT_DATA_PATH, TEST_FILE, TRAIN_FILE
from knowledge_distillation.bert_data import df_to_dataset
from knowledge_distillation.utils import get_logger

if __name__ == '__main__':

    logger = get_logger()

    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8')
    test_df = pd.read_csv(TEST_FILE, encoding='utf-8')

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = df_to_dataset(train_df, bert_tokenizer, bert_settings['max_seq_length'])
    test_dataset = df_to_dataset(test_df, bert_tokenizer, bert_settings['max_seq_length'])

    trainer = BertTrainer(bert_settings, logger)
    model = trainer.train(train_dataset, bert_tokenizer, ROOT_DATA_PATH)

    logger.info('validate on test dataset')
    _, acc = trainer.evaluate(model, test_dataset)
    logger.info('accuracy={:.4f}'.format(acc))
