# coding: utf-8

from __future__ import unicode_literals, print_function

import torch


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad(seq, max_len):
    if len(seq) < max_len:
        seq = seq + ['<pad>'] * (max_len - len(seq))

    return seq[0:max_len]


def to_indexes(vocab, words):
    return [vocab.stoi[w] for w in words]
