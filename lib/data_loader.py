import scipy
import scipy.linalg as la
import numpy as np
from scipy.io import loadmat
import torch


def shuffle_data(data_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(data_size)
    return np.random.permutation(indices)


def EA(data):
    """data alignment"""
    data_align = []
    length = len(data)
    rf_matrix = np.dot(data[0], np.transpose(data[0]))
    for i in range(1, length):
        rf_matrix += np.dot(data[i], np.transpose(data[i]))
    rf_matrix /= length

    rf = la.inv(la.sqrtm(rf_matrix))
    if rf.dtype == complex:
        rf = rf.astype(np.float64)

    for i in range(length):
        data_align.append(np.dot(rf, data[i]))

    return np.asarray(data_align)


def standard_normalize(x, clip_range=None):
    x = (x - np.mean(x)) / np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x


def load(data_path, uid, ratio=1.0, downsample=False, isEA=False):
    """ load data """
    data = loadmat(data_path + f's{uid}.mat')
    x = data['x']
    y = data['y']
    y = np.squeeze(y.flatten())
    # downsample
    if downsample:
        x1 = x[np.where(y == 0)]
        x2 = x[np.where(y == 1)]
        sample_num = min(len(x1), len(x2))
        idx1, idx2 = shuffle_data(len(x1)), shuffle_data(len(x2))
        x = np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
        y = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)
    if isEA:
        x = EA(x.squeeze())
    else:
        x = x.squeeze()

    x_sampled, y_sampled = [], []
    if 'RSVP' in data_path:
        for i in np.unique(y):
            x_, y_ = x[y == i], y[y == i]
            idx = np.random.permutation(np.arange(len(x_)))
            x_, y_ = x_[idx[:int(ratio*len(x_))]], y_[idx[:int(ratio*len(y_))]]
            x_sampled.append(x_)
            y_sampled.append(y_.reshape(-1, 1))
        x_sampled = np.concatenate(x_sampled, axis=0)
        y_sampled = np.squeeze(np.concatenate(y_sampled, axis=0))
    else:
        idx = np.random.permutation(np.arange(len(x)))
        x_sampled, y_sampled = x[idx[:int(ratio * len(x))]], y[idx[:int(ratio * len(y))]]
    
    return x_sampled, y_sampled


def balance_split(x, y, ratio):
    lb_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = np.random.choice(idx, int(np.ceil(len(idx) * ratio)), False)
        lb_idx.extend(idx)
    ulb_idx = np.array(sorted(list(set(range(len(x))) - set(lb_idx))))
    return lb_idx, ulb_idx


def random_split(x, y, ratio):
    idx = np.random.permutation(np.arange(len(x)))
    lb_idx = idx[:int(ratio * len(x))]
    while len(np.unique(y[idx])) != len(np.unique(y)):
        idx = np.random.permutation(np.arange(len(x)))
        lb_idx = idx[:int(ratio * len(x))]
    ulb_idx = np.array(sorted(list(set(range(len(x))) - set(lb_idx))))
    return lb_idx, ulb_idx


class GsNoise(object):
    def __init__(self, mean, alpha):
        self.mean = mean
        self.alpha = alpha

    def __call__(self, x):
        am = torch.mean(torch.std(x, dim=-1))
        x = x.clone().detach() + torch.empty_like(x).normal_(mean=self.mean, std=am * self.alpha)

        return x


class UniNoise(object):
    def __init__(self, am=0.5) -> None:
        super().__init__()
        self.am = am

    def __call__(self, input):
        rand_t = torch.rand(input.size()) - 0.5
        to_add_t = rand_t * 2 * self.am
        input = input + to_add_t
        return input


def BNCILoad(data_path, id, lb_ratio=0.05, align=''):
    label_dict_1 = {
        'left_hand': 0,
        'right_hand': 1,
    }
    label_dict_2 = {
        'left_hand': 0,
        'right_hand': 1,
        'feet': 2,
        'tongue': 3
    }
    label_dict_3 = {
        'right_hand': 0,
        'feet': 1
    }
    if 'BNCI2014-001-2' in data_path: label_dict = label_dict_1
    elif 'BNCI2014-001-4' in data_path: label_dict = label_dict_2
    elif 'BNCI2014-002-2' in data_path: label_dict = label_dict_3
    elif 'BNCI2015-001-2' in data_path: label_dict = label_dict_3
    data = scipy.io.loadmat(data_path + f'A{id + 1}.mat')
    x, y = data['X'], data['y']
    try:
        y = np.array([label_dict[y[j].replace(' ', '')] for j in range(len(y))]).reshape(-1)
    except TypeError or KeyError:
        y = y.reshape(-1)
    lb_idx, ulb_idx = random_split(x, y, ratio=lb_ratio)
    x_lb, y_lb = x[lb_idx], y[lb_idx]

    if align == 'EA':
        x_lb = EA(x_lb)
    return x_lb, y_lb


