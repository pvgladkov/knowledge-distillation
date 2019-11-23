# coding: utf-8

from __future__ import unicode_literals, print_function

import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from experiments.sst2.bert_trainer import batch_to_inputs
from experiments.sst2.lstm_trainer import LSTMDistilled
from experiments.sst2.settings import distillation_settings, TRAIN_FILE, TEST_FILE, ROOT_DATA_PATH
from knowledge_distillation.bert_data import df_to_dataset
from knowledge_distillation.utils import get_logger, device

if __name__ == '__main__':
    logger = get_logger()

    # 1. get data
    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8')
    test_df = pd.read_csv(TEST_FILE, encoding='utf-8')

    bert_model = BertForSequenceClassification.from_pretrained(ROOT_DATA_PATH)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = df_to_dataset(train_df, tokenizer, distillation_settings['max_seq_length'])
    sampler = SequentialSampler(train_dataset)
    data = DataLoader(train_dataset, sampler=sampler, batch_size=distillation_settings['train_batch_size'])

    bert_model.to(device())
    bert_model.eval()

    bert_logits = None

    for batch in tqdm(data, desc="bert logits"):
        batch = tuple(t.to(device()) for t in batch)
        inputs = batch_to_inputs(batch)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            _, logits = outputs[:2]

            logits = logits.cpu().numpy()
            if bert_logits is None:
                bert_logits = logits
            else:
                bert_logits = np.vstack((bert_logits, logits))

    # 2.
    X_train = train_df['text']
    y_train = bert_logits
    y_real = train_df['label']

    # 3. trainer
    distiller = LSTMDistilled(distillation_settings, logger)

    # 4. train
    model, vocab = distiller.train(X_train, y_train, y_real, ROOT_DATA_PATH)

    # 5. test
    X_test = test_df['text']
    y_test = test_df['label']
    distiller.validate(X_test, y_test, y_test, model, vocab)
