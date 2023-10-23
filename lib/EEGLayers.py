import torch
import torch.nn as nn


def CalculateOutSize(blocks, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    x = torch.rand(1, 1, channels, samples)
    for block in blocks:
        block.eval()
        x = block(x)
    shape = x.shape[-2] * x.shape[-1]
    return shape


class CSP(nn.Module):
    def __init__(self, filters, transform_into='csp_space', log=False):
        """
        CSP Layer
        :param filters: CSP spatial filters, (filters, channels).
        :param transform_into: 'csp_space' or 'average_power'.
        :param log: only needed when transform_into=='average_power'
        :param kwargs:
        """
        super(CSP, self).__init__()
        self.nb_filters = int(filters.shape[0])
        self.transform_into = transform_into
        self.log = log 
        self.csp_filters = nn.Parameter(filters)

    def forward(self, x):
        kernel = self.csp_filters.unsqueeze(0).tile((x.shape[0], 1, 1))
        x_ = torch.matmul(kernel, x)
        
        if self.transform_into == 'csp_space':
            return x_
        elif self.transform_into == 'average_power':
            power = torch.mean(torch.square(x_), dim=2, keepdim=False)
            if self.log: power = torch.log(power)
            return power
        else:
            raise Exception(f'{self.transform_into} is not valid!')


class BN(nn.Module):
    def __init__(self, num_features, mean, var):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features)
        self.bn.running_mean = mean
        self.bn.running_var = var
    
    def forward(self, x):
        return self.bn(x)


class CSP_BN(nn.Module):
    def __init__(self, num_features, mean, var, filters, transform_into='average_power', log=False):
        super().__init__()
        self.nets = nn.Sequential(CSP(filters, transform_into='average_power'),
                                  BN(num_features=num_features, mean=mean, var=var))

    def forward(self, x):
        return self.nets(x)


class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features, weight, bias):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)
    def forward(self, x):
        return self.linear(x)