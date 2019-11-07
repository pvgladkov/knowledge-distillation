# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd
from torchtext import data
from torchtext import datasets

if __name__ == '__main__':
    TEXT = data.Field(sequential=True, include_lengths=True, lower=True, init_token='<sos>', eos_token='<eos>')
    LABEL = data.LabelField()
    train, dev, test = datasets.SST.splits(TEXT, LABEL)

    train_text = [' '.join(t.text) for t in train.examples]
    test_text = [' '.join(t.text) for t in test.examples]

    train_label = [t.label for t in train.examples]
    test_label = [t.label for t in test.examples]

    train_df = pd.DataFrame({'text': train_text, 'label': train_label})
    test_df = pd.DataFrame({'text': test_text, 'label': test_label})

    filtered_train_df = train_df[train_df['label'] != 'neutral']
    filtered_test_df = test_df[test_df['label'] != 'neutral']

    filtered_train_df['label'] = filtered_train_df['label'].apply(lambda x: int(x == 'positive'))
    filtered_test_df['label'] = filtered_test_df['label'].apply(lambda x: int(x == 'positive'))

    filtered_train_df.to_csv('./.data/sst/train.csv', encoding='utf-8', index=False)
    filtered_test_df.to_csv('./.data/sst/test.csv', encoding='utf-8', index=False)

    print('ok')