# coding: utf-8

from __future__ import unicode_literals, print_function

import torch
from torch import nn


class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, batch_size, device=None):
        super(SimpleLSTM, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.LSTM(self.embedding.embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.device = self.init_device(device)

    @staticmethod
    def init_device(device):
        if device is None:
            return torch.device('cuda')
        return device

    def forward(self, text, text_lengths=None):

        x = self.embedding(text)

        x, hidden = self.rnn(x)

        hidden, cell = hidden
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        x = self.fc(hidden)
        return x
