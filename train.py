import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F

from dataloader import MRDataset, get_id_grade, get_label

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from importlib import import_module


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(task, model, train_loader, epoch, num_epochs, optimizer, current_lr, log_every=100):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    losses = []

    for i, (image, label, weight) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float())

        if task == 'multi':
            loss = torch.nn.CrossEntropyLoss(weight=weight)(prediction, label)
        else:
            loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)
        
        y_trues.append(label[0].tolist())
        y_preds.append(probas[0].tolist())

        try:
            auc = np.round(metrics.roc_auc_score(y_trues, y_preds), 4)
        except:
            auc = 0.5

        if i % log_every == 0 and i > 0:
            print(f"[Epoch: {epoch + 1} / {num_epochs} | Single batch number : {i} / {len(train_loader)}] | "
                  f"avg train loss {np.round(np.mean(losses), 4)} | train auc : {auc} | lr : {current_lr}")

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = auc
    return train_loss_epoch, train_auc_epoch


def validate_model(task, model, val_loader, epoch, num_epochs, current_lr, log_every=20):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []

    for i, (image, label, weight) in enumerate(val_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float())

        if task == 'multi':
            loss = torch.nn.CrossEntropyLoss(weight=weight)(prediction, label)
        else:
            loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(label[0].tolist())
        y_preds.append(probas[0].tolist())

        try:
            auc = np.round(metrics.roc_auc_score(y_trues, y_preds), 4)
        except:
            auc = 0.5

        if i % log_every == 0 and i > 0:
            print(f"[Epoch: {epoch + 1} / {num_epochs} | Single batch number : {i} / {len(val_loader)}] | "
                  f"avg train loss {np.round(np.mean(losses), 4)} | val auc : {auc} | lr : {current_lr}")

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = auc
    return val_loss_epoch, val_auc_epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run(args, BASE_PATH, LABEL_PATH):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
    
    augmentor = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        RandomFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])

    id_grade = get_id_grade(LABEL_PATH)
    file_paths = np.array([os.path.join(BASE_PATH, file) for file in os.listdir(BASE_PATH)])
    labels = np.array(get_label(args.task, file_paths, id_grade))

    X_train, _, y_train, _ = train_test_split(file_paths, labels, test_size=0.1, stratify=labels, random_state=42)

    k = 9
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        if fold == 5:
            break
        
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        train_dataset = MRDataset(args.task, X_train_fold, y_train_fold, transform=augmentor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=11, drop_last=False)

        validation_dataset = MRDataset(args.task, X_val_fold, y_val_fold)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)
        
        mrnet_class = getattr(import_module("model"), args.model)
        mrnet = mrnet_class(args.task)

        if torch.cuda.is_available():
            mrnet = mrnet.cuda()

        optimizer = optim.Adam(mrnet.parameters(), lr=args.lr, weight_decay=0.1)

        if args.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
        elif args.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=args.gamma)

        best_val_loss = float('inf')
        best_val_auc = float(0)

        num_epochs = args.epochs
        iteration_change_loss = 0
        patience = args.patience
        log_every = args.log_every

        for epoch in range(num_epochs):
            current_lr = get_lr(optimizer)

            t_start = time.time()
            
            train_loss, train_auc = train_model(
                args.task, mrnet, train_loader, epoch, num_epochs, optimizer, current_lr, log_every)
            val_loss, val_auc = validate_model(
                args.task, mrnet, validation_loader, epoch, num_epochs, current_lr)

            if args.lr_scheduler == 'plateau':
                scheduler.step(val_loss)
            elif args.lr_scheduler == 'step':
                scheduler.step()

            t_end = time.time()
            delta = t_end - t_start

            print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
                train_loss, train_auc, val_loss, val_auc, delta))

            iteration_change_loss += 1
            print('-' * 30)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                if bool(args.save_model):
                    file_name = f'{args.model}_{args.task}_{args.plane}_fold_{fold + 1}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
                    for f in os.listdir('./models/'):
                        if (args.model in f) and (args.task in f) and (args.plane in f) and (f"fold_{fold + 1}" in f):
                            os.remove(f'./models/{f}')
                    torch.save(mrnet, f'./models/{file_name}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                iteration_change_loss = 0

            if iteration_change_loss == patience:
                print('Early stopping after {0} iterations without the decrease of the val loss'.
                    format(iteration_change_loss))
                break


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['binary', 'multi'])
    parser.add_argument('-m', '--model', type=str, default='MRNet')
    parser.add_argument('--cuda', type=int, default=0, help="CUDA device number")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    BASE_PATH = f"DATASET_PATH"
    LABEL_PATH = f"LABEL_PATH"
    run(args)
