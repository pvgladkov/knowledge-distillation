# coding: utf-8

from __future__ import unicode_literals, print_function

import os

import torch
from torch.utils.data import TensorDataset, random_split, RandomSampler, DataLoader, SequentialSampler
from torchtext import data
from tqdm import tqdm

from knowledge_distillation.loss import WeightedMSE
from knowledge_distillation.modeling_lstm import SimpleLSTM
from knowledge_distillation.utils import device, to_indexes, pad
from experiments.sst2.utils import normalize


class _LSTMBase(object):

    vocab_name = None
    weights_name = None

    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def model(self, text_field):
        raise NotImplementedError()

    @staticmethod
    def to_dataset(x, y, y_real):
        torch_x = torch.tensor(x, dtype=torch.long)
        torch_y = torch.tensor(y, dtype=torch.float)
        torch_real_y = torch.tensor(y_real, dtype=torch.long)
        return TensorDataset(torch_x, torch_y, torch_real_y)

    @staticmethod
    def to_device(text, bert_prob, real_label):
        text = text.to(device())
        bert_prob = bert_prob.to(device())
        real_label = real_label.to(device())
        return text, bert_prob, real_label

    def train(self, X, y, y_real, output_dir):

        X_split = [normalize(t.split()) for t in X]

        text_field = data.Field()

        text_field.build_vocab(X_split, max_size=10000)

        # pad
        X_pad = [pad(s, self.settings['max_seq_length']) for s in tqdm(X_split, desc='pad')]

        # to index
        X_index = [to_indexes(text_field.vocab, s) for s in tqdm(X_pad, desc='to index')]

        dataset = self.to_dataset(X_index, y, y_real)
        val_len = int(len(dataset) * 0.1)
        train_dataset, val_dataset = random_split(dataset, (len(dataset) - val_len, val_len))

        model = self.model(text_field)
        model.to(device())

        self.full_train(model, train_dataset, val_dataset, output_dir)
        torch.save(text_field, os.path.join(output_dir, self.vocab_name))

        return model, text_field.vocab

    def validate(self, X, y, model, vocab):

        X_split = [normalize(t.split()) for t in X]

        # pad
        X_pad = [pad(s, self.settings['max_seq_length']) for s in tqdm(X_split, desc='pad')]

        # to index
        X_index = [to_indexes(vocab, s) for s in tqdm(X_pad, desc='to index')]

        dataset = self.to_dataset(X_index, y, y)
        _, acc = self.epoch_evaluate_func(model, dataset)
        self.logger.info('accuracy={}'.format(acc))

    @staticmethod
    def optimizer(model):
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
            eval_loss, acc = self.epoch_evaluate_func(model, val_dataset)

            self.logger.info('######## epoch={} ########'.format(epoch))
            self.logger.info("train_loss={:.4f}".format(train_loss))
            self.logger.info("val_loss={:.4f}".format(eval_loss))
            self.logger.info("val_acc={:.4f}".format(acc))

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.logger.info('save best model {:.4f}'.format(eval_loss))
                torch.save(model.state_dict(), os.path.join(output_dir, self.weights_name))

    def epoch_train_func(self, model, dataset):
        train_loss = 0
        train_sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=self.settings['train_batch_size'],
                                 drop_last=True)
        model.train()
        num_examples = 0
        optimizer, scheduler = self.optimizer(model)
        for i, (text, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Train')):
            text, bert_prob, real_label = self.to_device(text, bert_prob, real_label)
            model.zero_grad()
            output = model(text.t()).squeeze(1)
            loss = self.loss(output, bert_prob, real_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_examples += len(real_label)
        scheduler.step()
        return train_loss / num_examples

    def epoch_evaluate_func(self, model, eval_dataset):
        """
        :param model:
        :param eval_dataset:s
        :return:
        """
        eval_sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=self.settings['eval_batch_size'],
                                 drop_last=True)

        eval_loss = 0.0
        acc = 0.0
        num_examples = 0
        model.eval()
        for i, (text, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Val')):
            text, bert_prob, real_label = self.to_device(text, bert_prob, real_label)
            output = model(text.t()).squeeze(1)
            loss = self.loss(output, bert_prob, real_label)
            eval_loss += loss.item()

            pred_label = torch.argmax(output, dim=1)
            acc += torch.sum(pred_label == real_label).cpu().numpy()
            num_examples += len(real_label)

        return eval_loss / num_examples, acc / num_examples

    def loss(self, output, bert_prob, real_label):
        raise NotImplementedError()


class LSTMBaseline(_LSTMBase):
    """
    LSTM baseline
    """

    vocab_name = 'text_vocab.pt'
    weights_name = 'simple_lstm.pt'

    def __init__(self, settings, logger):
        super(LSTMBaseline, self).__init__(settings, logger)
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss(self, output, bert_prob, real_label):
        return self.criterion(output, real_label)

    def model(self, text_field):
        model = SimpleLSTM(
            input_dim=len(text_field.vocab),
            embedding_dim=64,
            hidden_dim=128,
            output_dim=2,
            n_layers=1,
            bidirectional=True,
            dropout=0.5,
            batch_size=self.settings['train_batch_size'])
        return model


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

    def model(self, text_field):
        model = SimpleLSTM(
            input_dim=len(text_field.vocab),
            embedding_dim=64,
            hidden_dim=128,
            output_dim=2,
            n_layers=1,
            bidirectional=True,
            dropout=0.5,
            batch_size=self.settings['train_batch_size'])
        return model


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
