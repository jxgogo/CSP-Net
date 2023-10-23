import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from lib.data_loader import BNCILoad
from lib.models import EEGNet, Classifier, CalculateOutSize, ShallowConvNet, DeepConvNet
from lib.pytorch_utils import seed, split_data, bca_score, print_args, init_weights
import numpy as np
import os
import logging
import torch
import argparse

from base_trainer import csp_trainer

build_model = {
    'CSP': csp_trainer.Trainer,
}

def train(x_train, y_train, x_test, y_test, args):
    # initialize the model
    trainer = build_model[args.alg](args)
    ori_acc, ori_bca = trainer.ori_train(x_train, y_train, x_test, y_test)

    logging.info(f'origin {i}: acc-{ori_acc} bca-{ori_bca}')

    # ================================ RETRAIN =========================
    filters = torch.from_numpy(np.array(trainer.csp.filters_[:trainer.csp.n_components])).type(
        torch.FloatTensor)

    if args.baseline == 2:
        filters = torch.nn.init.kaiming_uniform_(filters)

    if args.model == 'EEGNet':
        modelF = EEGNet(Chans=x_train.shape[-2],
                        Samples=x_train.shape[-1],
                        kernLenght=64,
                        F1=4,
                        D=2,
                        F2=8,
                        dropoutRate=0.25,
                        filters=filters if args.baseline != 1 else None).to(args.device)
    elif args.model == 'ShallowCNN':
        modelF = ShallowConvNet(Chans=x_train.shape[-2],
                                Samples=x_train.shape[-1],
                                dropoutRate=0.5,
                                filters=filters if args.baseline != 1 else None).to(args.device)
    elif args.model == 'DeepCNN':
        modelF = DeepConvNet(Chans=x_train.shape[-2],
                             Samples=x_train.shape[-1],
                             dropoutRate=0.5,
                             filters=filters if args.baseline != 1 else None).to(args.device)
    embed_dim = CalculateOutSize(modelF, x_train.shape[-2], x_train.shape[-1])
    modelC = Classifier(embed_dim, args.classes).to(args.device)
    modelF.apply(init_weights)
    modelC.apply(init_weights)

    if args.baseline == 3 or args.baseline == 0:
        modelF.csp_filters.requires_grad = False

    # trainable parameters
    params = []
    for name, v in modelF.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for name, v in modelC.named_parameters():
         params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # data loader
    x_train = Variable(torch.from_numpy(x_train).type(torch.FloatTensor))
    y_train = Variable(torch.from_numpy(y_train).type(torch.LongTensor))

    x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
    y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))
    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False)
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)

    hists = []
    best_acc, best_bca = 0, 0
    for epoch in range(args.epochs):
        # model training
        if args.baseline == 0 and epoch >= 50:
            modelF.csp_filters.requires_grad = True
        modelF.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            out = modelC(modelF(batch_x))
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            modelF.MaxNormConstraint()
            modelC.MaxNormConstraint()

        if (epoch + 1) % 1 == 0:
            modelF.eval()
            train_loss, train_acc, train_bca = eval(modelF, modelC, criterion,
                                                    train_loader, args)
            test_loss, test_acc, test_bca = eval(modelF, modelC, criterion, test_loader,
                                                 args)
            if test_acc >= best_acc:
                best_acc = test_acc
                best_bca = test_bca

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} train bca: {:.2f}| test acc: {:.2f} test bca: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc,
                        train_bca, test_acc, test_bca))

            hists.append({
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "best_bca": best_bca,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_bca": train_bca,
                "loss": test_loss,
                "acc": test_acc,
                "bca": test_bca,
                "ori_acc": ori_acc,
                "ori_bca": ori_bca
            })
    modelF.eval()
    test_loss, test_acc, test_bca = eval(modelF, modelC, criterion, test_loader, args)
    logging.info(f'test acc: {test_acc} bca: {test_bca}')

    return hists


def eval(modelF: nn.Module, modelC: nn.Module, criterion: nn.Module, data_loader: DataLoader, args):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = modelC(modelF(x))
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(labels)
    acc = correct / len(labels)
    bca = bca_score(labels, preds)

    return loss, acc, bca


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='2')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dataset', type=str, default='BNCI_MI4C')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--align', default='EA')

    parser.add_argument('--alg', type=str, default='CSP', choices=['CSP'])
    parser.add_argument('--filters', type=int, default=8, help='CSP parameter')

    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--setting', type=str, default='cross')
    parser.add_argument('--baseline', type=int, default=0, help='0-输入层filter并retrain； 1-原始EEGNet；2-输入层随机filter；3-输入层filter不retrain')
    parser.add_argument('--log', type=str, default='')
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    subject_num = {'BNCI_MI4C': 9, 'BNCI_MI2C': 9, 'BNCI_MI14S': 14, 'BNCI_MI9S': 9}
    data_path = {'BNCI_MI4C': '../dataset/MIDataNew/BNCI2014-001-4/',
                 'BNCI_MI2C': '../dataset/MIDataNew/BNCI2014-001-2/',
                 'BNCI_MI14S': '../dataset/MIDataNew/BNCI2014-002-2/',
                 'BNCI_MI9S': '../dataset/MIDataNew/BNCI2015-001-2/'}
    classes = {'BNCI_MI4C': 4, 'BNCI_MI2C': 2, 'BNCI_MI14S': 2, 'BNCI_MI9S': 2}
    channels = {'BNCI_MI4C': 22, 'BNCI_MI2C': 22, 'BNCI_MI14S': 15, 'BNCI_MI9S': 13}
    args.path, args.classes, args.channel = data_path[args.dataset], classes[args.dataset], channels[args.dataset]

    # path build
    log_name = f'{args.setting}_baseline{args.baseline}' if not len(
        args.log) else f'{args.setting}_baseline{args.baseline}' + f'_{args.log}'
    results_path = os.path.join(f'results_{args.alg}_{args.model}', 'Process', args.dataset, str(args.ratio))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(os.path.join(results_path, log_name + '.log'), mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')

    # Build pathes
    r_bcas = []
    r_accs = []
    r_re_bcas = []
    r_re_accs = []
    import pandas as pd
    dfs = pd.DataFrame()
    for r in range(5):
        seed(r*10)
        bcas = []
        accs = []
        re_bcas = []
        re_accs = []
        for i in range(subject_num[args.dataset]):
            if args.setting == 'within':
                x_train, y_train = BNCILoad(args.path, id=i, lb_ratio=1.0, align='')
                x_train, y_train, x_test, y_test, _, _ = split_data([x_train, y_train], split=0.8, shuffle=True)
                x_train, y_train, _, _, _, _ = split_data([x_train, y_train], split=args.ratio, shuffle=True)
            else:
                x_train, y_train = [], []
                for j in range(subject_num[args.dataset]):
                    if i == j:
                        x_test, y_test = BNCILoad(args.path, id=j, lb_ratio=1.0,
                                                  align='EA' if args.align == 'EA' else '')
                    else:
                        x_, y_ = BNCILoad(args.path, id=j, lb_ratio=args.ratio,
                                          align='EA' if args.align == 'EA' else '')
                        x_train.append(x_)
                        y_train.append(y_)
                x_train = np.concatenate(x_train, axis=0)
                y_train = np.concatenate(y_train, axis=0)
            data_size = y_train.shape[0]

            hists = train(x_train, y_train, x_test, y_test, args)
            accs.append(hists[-1]['ori_acc'])
            bcas.append(hists[-1]['ori_bca'])
            re_accs.append(hists[-1]['acc'])
            re_bcas.append(hists[-1]['bca'])

            df = pd.DataFrame(hists)
            df["method"] = [log_name] * len(hists)
            df["rep"] = [r] * len(hists)
            df["s"] = [i] * len(hists)
            dfs = pd.concat([dfs, df], axis=0)

        logging.info('*' * 200)
        logging.info(f'repeat {r}, or: mean_acc:{np.mean(accs)} mean_bca:{np.mean(bcas)}')
        logging.info(f'repeat {r}, re: mean_acc:{np.mean(re_accs)} mean_bca:{np.mean(re_bcas)}')
        np.savez(os.path.join(results_path, f'z_{log_name}_r{str(r)}_results.npz'),
                 acc=np.array(re_accs), bca=np.array(re_bcas),
                 ori_acc=np.array(accs), ori_bca=np.array(bcas))
        r_accs.append(accs)
        r_bcas.append(bcas)
        r_re_accs.append(re_accs)
        r_re_bcas.append(re_bcas)

    # log
    logging.info(
        '----------------------------------- Baseline results -----------------------------------'
    )
    logging.info(f'acc: {r_accs}')
    logging.info(f'bca: {r_bcas}')
    logging.info(f'Mean -- acc on subjects: {np.mean(r_accs, axis=0)}, bca: {np.mean(r_bcas, axis=0)}')
    logging.info(f'Std -- acc on subjects: {np.std(r_accs, axis=0)}, bca: {np.std(r_bcas, axis=0)}')

    logging.info(
        '----------------------------------- Retrain results -----------------------------------'
    )
    logging.info(f're_acc: {r_re_accs}')
    logging.info(f're_bca: {r_re_bcas}')
    logging.info(f'Mean -- acc on subjects: {np.mean(r_re_accs, axis=0)}, bca: {np.mean(r_re_bcas, axis=0)}')
    logging.info(f'Std -- acc on subjects: {np.std(r_re_accs, axis=0)}, bca: {np.std(r_re_bcas, axis=0)}')

    logging.info(f'Baseline Mean acc: {np.mean(r_accs)} Mean bca: {np.mean(r_bcas)}')
    logging.info(f'Baseline Std acc: {np.std(np.mean(r_accs, axis=1))} Std bca: {np.std(np.mean(r_bcas, axis=1))}')
    logging.info(f'Retrain Mean acc: {np.mean(r_re_accs)} Mean bca: {np.mean(r_re_bcas)}')
    logging.info(f'Retrain Std acc: {np.std(np.mean(r_re_accs, axis=1))} Std bca: {np.std(np.mean(r_re_bcas, axis=1))}')

    # csv
    dfs.to_csv(os.path.join(results_path, log_name + '_raw.csv'), index=False)
    dfs_temp = dfs.groupby(by=["method", "rep", "epoch"]).mean().reset_index()
    avg_dfs = dfs_temp.groupby(by=["method", "epoch"]).mean().reset_index()
    avg_dfs = avg_dfs.sort_values(by=["method", "epoch"])
    avg_dfs["s"] = ["avg"] * len(avg_dfs)
    std_dfs = dfs_temp.groupby(by=["method", "epoch"]).std().reset_index()
    std_dfs = std_dfs.sort_values(by=["method", "epoch"])
    std_dfs["s"] = ["std"] * len(std_dfs)
    dfs = pd.concat([avg_dfs, std_dfs], axis=0)
    dfs = dfs.drop("rep", axis=1)
    dfs.to_csv(os.path.join(results_path, log_name+'_avg.csv'), index=False)

