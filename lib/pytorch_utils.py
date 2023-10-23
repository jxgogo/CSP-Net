import argparse
import random
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Optional

import torch
import torch.nn as nn


def print_args(args: argparse.ArgumentParser):
    """
    print the hyperparameters
    :param args: hyperparameters
    :return: None
    """
    s = "=========================================================\n"
    for arg, concent in args.__dict__.items():
        s += "{}:{}\n".format(arg, concent)
    return s


def init_weights(model: nn.Module):
    """
    Network Parameters Initialization Function
    :param model: the model to initialize
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(model.weight)


def bca_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def standard_normalize(x_train, x_test, clip_range=None):
    mean, std = np.mean(x_train), np.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    if clip_range is not None:
        x = np.clip(x_train, a_min=clip_range[0], a_max=clip_range[1])
        x = np.clip(x_test, a_min=clip_range[0], a_max=clip_range[1])
    return x_train, x_test


def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer: nn.Module, epoch: int,
                         learning_rate: float):
    """decrease the learning rate"""
    lr = learning_rate
    if epoch >= 50:
        lr = learning_rate * 0.1
    if epoch >= 100:
        lr = learning_rate * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_for_balanced_classes(y: torch.Tensor):
    count = [0.0] * len(np.unique(y.numpy()))
    for label in y:
        count[label] += 1.0
    count = [len(y) / x for x in count]
    weight = [0.0] * len(y)
    for idx, label in enumerate(y):
        weight[idx] = count[label]

    return weight


def split_data_benchmark(data, all_bloack, split_block):
    x = data[0]
    y = data[1]
    indices = np.random.permutation(np.arange(all_bloack))
    x_train = x[indices[:split_block], :, :, :]
    y_train = y[indices[:split_block]]
    x_test  = x[indices[split_block:], :, :, :]
    y_test  = y[indices[split_block:]]

    return x_train, y_train, x_test, y_test


def split_data(data, split=0.8, shuffle=True, downsample=False):
    x = data[0]
    y = data[1]

    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)

    train_idx = indices[:split_index]
    test_idx  = indices[split_index:]
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    if downsample:
        x_train1 = x_train[np.where(y_train == 0)]
        x_train2 = x_train[np.where(y_train == 1)]
        sample_num = min(len(x_train1), len(x_train2))
        idx1, idx2 = np.random.permutation(np.arange(len(x_train1))), np.random.permutation(np.arange(len(x_train2)))
        x_train = np.concatenate([x_train1[idx1[:sample_num]], x_train2[idx2[:sample_num]]], axis=0)
        y_train = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)

    return x_train, y_train, x_test, y_test, train_idx, test_idx


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class GeneralTorchModel(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.eval()
        self.num_queries = 0
        self.n_class = n_class

    def forward(self, image):
        logits = self.model(image)
        return logits


    def predict_prob(self, image):
        with torch.no_grad():
            logits = self.model(image)
            self.num_queries += image.size(0)
        return logits

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict
