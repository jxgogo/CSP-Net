import torch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP as mne_CSP

from lib import EEGLayers
from lib.pytorch_utils import bca_score


class Trainer:
    def __init__(self, args):
        self.csp = mne_CSP(n_components=args.filters, transform_into='average_power', log=False, cov_est='epoch')
        self.lr = LogisticRegression(solver='sag', max_iter=200, C=0.01, multi_class='multinomial')
        self.model = Pipeline([('csp_power', self.csp), ('LR', self.lr)])
        self.baseline = args.baseline
        self.device = args.device
        self.filters = args.filters
        self.feat_dim = args.filters
        self.classes = args.classes
        self.learning_rate = args.lr

    def ori_train(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train)
        y_pred = np.argmax(self.model.predict_proba(x_test), axis=1)
        acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
        bca = bca_score(y_test, y_pred)
        return acc, bca

    def retrain_net(self):
        filters = torch.from_numpy(np.array(self.csp.filters_[:self.csp.n_components])).type(torch.FloatTensor)
        mean = torch.from_numpy(self.csp.mean_).type(torch.FloatTensor)
        var = torch.from_numpy(np.square(self.csp.std_)).type(torch.FloatTensor)
        coef = self.lr.coef_.astype(np.float32)
        intercept_ = self.lr.intercept_.astype(np.float32)
        if len(intercept_) == 1:
            coef = np.concatenate((-coef, coef), axis=0)
            intercept_ = np.concatenate((-intercept_, intercept_), axis=0)
        w = torch.from_numpy(np.array(coef)).type(torch.FloatTensor)
        b = torch.from_numpy(np.array(intercept_)).type(torch.FloatTensor)

        if self.baseline == 1:
            filters = torch.nn.init.normal_(filters)
            mean = torch.nn.init.normal_(mean)
            var = torch.nn.init.normal_(var)
            w = torch.nn.init.normal_(w)
            b = torch.nn.init.normal_(b)

        CSP_layer = EEGLayers.CSP_BN(self.filters, mean, var, filters, transform_into='average_power', log=False).to(self.device)
        LR_layer = EEGLayers.LogisticRegression(in_features=self.feat_dim, out_features=self.classes, weight=w, bias=b).to(self.device)

        # trainable parameters
        params = []
        if self.baseline != 2:
            for _, v in CSP_layer.named_parameters():
                params += [{'params': v, 'lr': self.learning_rate}]
        for _, v in LR_layer.named_parameters():
            params += [{'params': v, 'lr': self.learning_rate * 1e-3}]

        return CSP_layer, LR_layer, params