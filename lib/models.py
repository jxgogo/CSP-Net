import torch
import torch.nn as nn
from typing import Optional


def CalculateOutSize(model, Chans, Samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(2, 1, Chans, Samples).to(device)
    model.eval()
    out = model(x)
    return out.shape[-1]


def LoadModel(model_name, Classes, Chans, Samples):
    if model_name == 'EEGNet':
        modelF = EEGNet(Chans=Chans,
                       Samples=Samples,
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25)
    elif model_name == 'DeepCNN':
        modelF = DeepConvNet(Chans=Chans, Samples=Samples, F1=25, dropoutRate=0.5)
    elif model_name == 'ShallowCNN':
        modelF = ShallowConvNet(Chans=Chans, Samples=Samples, dropoutRate=0.5)
    else:
        raise 'No such model'
    embed_dim = CalculateOutSize(modelF, Chans, Samples)
    modelC = Classifier(embed_dim, Classes)
    modelD = DomainDiscriminator(embed_dim, hidden_dim=128)
    return modelF, modelC, modelD, embed_dim


class EEGNet(nn.Module):
    """
    :param
    """
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: Optional[float] = 0.5,
                 filters=None):
        super(EEGNet, self).__init__()

        self.Chans = len(filters) if filters != None else Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.csp_filters = nn.Parameter(filters.unsqueeze(0)) if filters != None else None

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.csp_filters != None:
            x = torch.matmul(self.csp_filters, x.squeeze())
            x = x.unsqueeze(1)
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)


class DeepConvNet(nn.Module):
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 F1: Optional[int] = 25,
                 dropoutRate: Optional[float] = 0.5,
                 filters=None):
        super(DeepConvNet, self).__init__()

        self.Chans = len(filters) if filters != None else Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.csp_filters = nn.Parameter(filters.unsqueeze(0)) if filters != None else None

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=F1, out_channels=F1, kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=F1), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1*2, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=F1*2), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1*2, out_channels=F1*4, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=F1*4), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.csp_filters != None:
            x = torch.matmul(self.csp_filters, x.squeeze())
            x = x.unsqueeze(1)
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        Chans: int,
        Samples: int,
        F1: Optional[int] = 40,
        dropoutRate: Optional[float] = 0.5,
        filters=None
    ):
        super(ShallowConvNet, self).__init__()

        self.Chans = len(filters) if filters != None else Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.csp_filters = nn.Parameter(filters.unsqueeze(0)) if filters != None else None

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=F1,
                      out_channels=F1,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=F1),
            nn.ELU(),  #Activation('square'),  #
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            nn.ELU(),  #Activation('log'),   #
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.csp_filters != None:
            x = torch.matmul(self.csp_filters, x.squeeze())
            x = x.unsqueeze(1)
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim,
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator module - 2 layers MLP

    Parameters:
        - input_dim (int): dim of input features
        - hidden_dim (int): dim of hidden features
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super(DomainDiscriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)