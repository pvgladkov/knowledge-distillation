# coding: utf-8

from __future__ import unicode_literals, print_function

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, random_split)
from tqdm import tqdm
from transformers import AdamW
from transformers import BertForSequenceClassification, BertTokenizer


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_to_inputs(batch):
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'token_type_ids': batch[2],
              'labels': batch[3]}

    return inputs


class BertTrainer(object):

    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def train(self, train_dataset, tokenizer, output_dir):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        model.to(device())
        self.full_train(train_dataset, model, tokenizer, output_dir)
        return model

    def optimizer(self, model):

        train_settings = self.settings

        optimizer = AdamW(model.parameters(), lr=train_settings['learning_rate'],
                          eps=train_settings['adam_epsilon'])
        return optimizer

    def full_train(self, dataset, model, tokenizer, output_dir):

        train_settings = self.settings

        train_batch_size = train_settings['train_batch_size']
        num_train_epochs = train_settings['num_train_epochs']

        # val_len = int(len(dataset) * 0.1)
        val_len = 0
        train_dataset, val_dataset = random_split(dataset, (len(dataset)-val_len, val_len))

        optimizer = self.optimizer(model)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

        model.zero_grad()

        for epoch in range(int(num_train_epochs)):

            train_loss = self.epoch_train_func(model, train_dataset, train_batch_size, optimizer, scheduler)
            self.logger.info("epoch={}, train_loss={:.4f}".format(epoch, train_loss))
            model.save_pretrained(output_dir)

    def epoch_train_func(self, model, dataset, batch_size, optimizer, scheduler,):
        train_loss = 0.0

        train_sampler = RandomSampler(dataset)
        data = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

        for step, batch in enumerate(tqdm(data, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device()) for t in batch)
            inputs = batch_to_inputs(batch)
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            model.zero_grad()

        scheduler.step()
        return train_loss / len(data)

    def evaluate(self, model, eval_dataset):
        train_settings = self.settings

        eval_sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=train_settings['eval_batch_size'])

        eval_loss = 0.0
        acc = 0.0
        num_examples = 0
        for batch in tqdm(data_loader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device()) for t in batch)

            with torch.no_grad():
                inputs = batch_to_inputs(batch)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.item()

                pred_label = torch.argmax(logits, dim=1)
                acc += torch.sum(pred_label == batch[3]).cpu().numpy()
                num_examples += len(batch[3])

        return eval_loss / len(data_loader), acc / num_examples

