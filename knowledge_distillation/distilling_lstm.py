# coding: utf-8

from __future__ import unicode_literals, print_function

import os

import torch
from torch.utils.data import TensorDataset, random_split, RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm

from knowledge_distillation.loss import WeightedMSE
from knowledge_distillation.utils import device, to_indexes, pad
from knowledge_distillation.modeling_lstm import SimpleLSTM
from torchtext import data


class _LSTMBase(object):

    vocab_name = None
    weights_name = None

    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def model(self, TEXT):
        raise NotImplementedError()

    def to_dataset(self, x, x_lengths, y, y_real):
        raise NotImplementedError()

    @staticmethod
    def to_device(text, text_len, bert_prob, real_label):
        text, text_len = text.to(device()), text_len.to(device())
        bert_prob = bert_prob.to(device())
        real_label = real_label.to(device())
        return text, text_len, bert_prob, real_label

    def train(self, X, y, y_real, output_dir):

        max_len = self.settings['max_seq_length']

        X_split = [t.split() for t in X]

        TEXT = data.Field()

        TEXT.build_vocab(X_split, max_size=10000)

        # len
        X_lengths = [len(s) for s in tqdm(X_split, desc='lengths')]

        # pad
        X_pad = [pad(s, max_len) for s in tqdm(X_split, desc='pad')]

        # to index
        X_index = [to_indexes(TEXT.vocab, s) for s in tqdm(X_pad, desc='to index')]

        dataset = self.to_dataset(X_index, X_lengths, y, y_real)
        val_len = int(len(dataset) * 0.1)
        train_dataset, val_dataset = random_split(dataset, (len(dataset) - val_len, val_len))

        model = self.model(TEXT)
        model.to(device())

        self.full_train(model, train_dataset, val_dataset, output_dir)

        torch.save(TEXT, os.path.join(output_dir, self.vocab_name))

    def optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)
        return optimizer, scheduler

    def full_train(self, model, train_dataset, val_dataset, output_dir):
        """
        :param model:
        :param train_dataset:
        :param val_dataset:
        :param output_dir:
        :return:
        """
        train_settings = self.settings
        num_train_epochs = train_settings['num_train_epochs']

        best_eval_loss = 100000

        for epoch in range(num_train_epochs):
            train_loss = self.epoch_train_func(model, train_dataset)
            eval_loss = self.epoch_evaluate_func(model, val_dataset)

            self.logger.info("epoch={}, train_loss={}".format(epoch, train_loss))
            self.logger.info("epoch={}, val_loss={}".format(epoch, eval_loss))

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.logger.info('save best model {}'.format(eval_loss))
                torch.save(model.state_dict(), os.path.join(output_dir, self.weights_name))

    def epoch_train_func(self, model, dataset):
        raise NotImplementedError()

    def epoch_evaluate_func(self, model, eval_dataset):
        raise NotImplementedError()


class LSTMBaseline(_LSTMBase):
    """
    LSTM baseline
    """

    vocab_name = 'text_vocab.pt'
    weights_name = 'simple_lstm.pt'

    def __init__(self, settings, logger):
        super(LSTMBaseline, self).__init__(settings, logger)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(device())

    def model(self, TEXT):
        model = SimpleLSTM(len(TEXT.vocab), 64, 128, 1, 1, True, 0.5,
                           batch_size=self.settings['train_batch_size'])
        return model

    def to_dataset(self, x, x_lengths, y, y_real):
        torch_x = torch.tensor(x, dtype=torch.long)
        torch_lengths = torch.tensor(x_lengths, dtype=torch.long)
        torch_y = torch.tensor(y, dtype=torch.float)
        torch_real_y = torch.tensor(y_real, dtype=torch.float)
        return TensorDataset(torch_x, torch_lengths, torch_y, torch_real_y)

    def epoch_train_func(self, model, dataset):
        """
        Одна эпоха
        """
        train_loss = 0
        batch_size = self.settings['train_batch_size']
        train_sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        model.train()

        optimizer, scheduler = self.optimizer(model)

        for i, (text, text_len, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Train')):

            text, text_len, bert_prob, real_label = self.to_device(text, text_len, bert_prob, real_label)

            model.zero_grad()
            output = model(text.t(), text_len).squeeze(1)

            loss = self.criterion(output, real_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        return train_loss / len(data_loader)

    def epoch_evaluate_func(self, model, eval_dataset):
        """
        :param model:
        :param eval_dataset:s
        :return:
        """
        batch_size = self.settings['eval_batch_size']
        eval_sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, drop_last=True)

        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num examples = %d", len(eval_dataset))
        self.logger.info("  Batch size = %d", batch_size)
        eval_loss = 0.0
        model.eval()
        for i, (text, text_len, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Val')):

            text, text_len, bert_prob, real_label = self.to_device(text, text_len, bert_prob, real_label)

            output = model(text.t(), text_len).squeeze(1)
            loss = self.criterion(output, real_label)
            eval_loss += loss.item()

        return eval_loss / len(data_loader)


class LSTMDistilled(_LSTMBase):
    """
    LSTM distilled
    """

    vocab_name = 'distil_text_vocab.pt'
    weights_name = 'distil_lstm.pt'

    def __init__(self, settings, logger):
        super(LSTMDistilled, self).__init__(settings, logger)
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.a = 0.5

    def loss(self, output, bert_prob, real_label):
        return self.a * self.criterion_ce(output, real_label) + (1 - self.a) * self.criterion_mse(output, bert_prob)

    def model(self, TEXT):
        model = SimpleLSTM(len(TEXT.vocab), 64, 128, 2, 1, True, 0.5,
                           batch_size=self.settings['train_batch_size'])
        return model

    def to_dataset(self, x, x_lengths, y, y_real):
        torch_x = torch.tensor(x, dtype=torch.long)
        torch_lengths = torch.tensor(x_lengths, dtype=torch.long)
        torch_y = torch.tensor(y, dtype=torch.float)
        torch_real_y = torch.tensor(y_real, dtype=torch.long)
        return TensorDataset(torch_x, torch_lengths, torch_y, torch_real_y)

    def epoch_train_func(self, model, dataset):
        train_loss = 0
        batch_size = self.settings['train_batch_size']
        train_sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        model.train()

        optimizer, scheduler = self.optimizer(model)

        for i, (text, text_len, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Train')):

            model.zero_grad()
            text, text_len, bert_prob, real_label = self.to_device(text, text_len, bert_prob, real_label)

            output = model(text.t(), text_len).squeeze(1)

            loss = self.loss(output, bert_prob, real_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        return train_loss / len(data_loader)

    def epoch_evaluate_func(self, model, eval_dataset):
        batch_size = self.settings['eval_batch_size']
        eval_sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, drop_last=True)

        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num examples = %d", len(eval_dataset))
        self.logger.info("  Batch size = %d", batch_size)
        eval_loss = 0.0

        model.eval()
        for i, (text, text_len, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Val')):
            text, text_len, bert_prob, real_label = self.to_device(text, text_len, bert_prob, real_label)

            output = model(text.t(), text_len).squeeze(1)
            loss = self.loss(output, bert_prob, real_label)
            eval_loss += loss.item()

        return eval_loss / len(data_loader)


class LSTMDistilledWeighted(LSTMDistilled):
    """
    LSTM distilled with weighted MSE
    """
    vocab_name = 'w_distil_text_vocab.pt'
    weights_name = 'w_distil_lstm.pt'

    def __init__(self, settings, logger):
        super(LSTMDistilledWeighted, self).__init__(settings, logger)
        self.criterion_mse = WeightedMSE()

    def loss(self, output, bert_prob, real_label):
        l1 = self.a * self.criterion_ce(output, real_label)
        l2 = (1 - self.a) * self.criterion_mse(output, bert_prob, real_label)
        return l1 + l2
