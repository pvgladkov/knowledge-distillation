# coding: utf-8

from __future__ import unicode_literals, print_function
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer

from knowledge_distillation.distilling_lstm import LSTMDistilled, LSTMDistilledWeighted
from knowledge_distillation.utils import get_logger
from experiments.sst2.train_bert import df_to_dataset
from experiments.sst2.bert_trainer import batch_to_inputs
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
import torch


settings = {
    'max_seq_length': 128,
    'num_train_epochs': 10,
    'train_batch_size': 32,
    'eval_batch_size': 32,
}


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    logger = get_logger()

    # 1. get data
    train_df = pd.read_csv('./.data/sst/train.csv', encoding='utf-8')
    test_df = pd.read_csv('./.data/sst/test.csv', encoding='utf-8')

    bert_model = BertForSequenceClassification.from_pretrained('/app/.data/sst')
    tokenizer = BertTokenizer.from_pretrained('/app/.data/sst')

    train_dataset = df_to_dataset(train_df, tokenizer)
    sampler = SequentialSampler(train_dataset)
    data = DataLoader(train_dataset, sampler=sampler, batch_size=settings['train_batch_size'])

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
    distiller = LSTMDistilledWeighted(settings, logger)

    # 4. train
    model, vocab = distiller.train(X_train, y_train, y_real, '/app/.data/sst')

    # 5. test
    X_test = test_df['text']
    y_test = test_df['label']
    distiller.validate(X_test, y_test, y_test, model, vocab)
