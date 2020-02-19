# coding: utf-8

from __future__ import unicode_literals, print_function

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
import numpy as np


class Trainer(object):

    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger
        self.tb_writer = SummaryWriter(log_dir=settings['log_dir'], filename_suffix=settings['tb_suffix'])

    def log_epoch(self, train_loss, eval_loss, acc, epoch):
        self.logger.info('######## epoch={} ########'.format(epoch))
        self.logger.info("train_loss={:.4f}, val_loss={:.4f}, acc={:.4f}".format(train_loss, eval_loss, acc))

        self.tb_writer.add_scalars('{}/loss'.format(self.settings['tb_suffix']),
                                   {'train': train_loss, 'test': eval_loss}, epoch)
        self.tb_writer.add_scalar('{}/val_acc'.format(self.settings['tb_suffix']), acc, epoch)

    def log_pr(self, labels, predictions, epoch):
        self.tb_writer.add_pr_curve('{}/pr'.format(self.settings['tb_suffix']), labels, predictions,
                                    global_step=epoch)
        self.tb_writer.add_scalar('{}/ap_score'.format(self.settings['tb_suffix']),
                                  average_precision_score(labels, predictions), global_step=epoch)

    @staticmethod
    def stack(predictions, labels, step_predictions, step_labels):
        step_predictions = step_predictions.detach().cpu().numpy()[:, 1]
        step_labels = step_labels.detach().cpu().numpy()
        if predictions is None:
            predictions = step_predictions
            labels = step_labels
        else:
            predictions = np.hstack((predictions, step_predictions))
            labels = np.hstack((labels, step_labels))

        return predictions, labels
