# coding: utf-8

from __future__ import unicode_literals, print_function

import torch
from torch import nn
from torch.nn.functional import log_softmax


class WeightedMSE(nn.Module):

    def forward(self, model_logits, bert_logits, labels):
        k = 1.5
        p = log_softmax(bert_logits, 1)
        ohe_labels = torch.nn.functional.one_hot(labels)
        nn_l = torch.sum(ohe_labels.float() * p, dim=1)
        weight = torch.exp(k * nn_l)
        return torch.mean(weight * torch.sum((model_logits - bert_logits) ** 2, dim=1))
