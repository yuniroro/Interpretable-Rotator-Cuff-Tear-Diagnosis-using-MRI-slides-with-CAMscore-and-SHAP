import torch
import numpy as np
import os

import argparse
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from dataloader import MRDataset, get_id_grade, get_label


def calculate_class_auc(y_true, y_score, classes):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    y_true_binarized = label_binarize(y_true, classes=classes)

    auc_scores = []
    for i in range(len(classes)):
        auc_scores.append(metrics.roc_auc_score(y_true_binarized[:, i], y_score[:, i]))
    return auc_scores


def calculate_class_accuracy(y_true, y_pred, classes):
    cfm = metrics.confusion_matrix(y_true, y_pred, labels=classes)
    class_accuracies = []

    for cls in classes:
        TP = cfm[cls, cls]
        TN = np.sum(cfm) - (np.sum(cfm[cls, :]) + np.sum(cfm[:, cls]) - TP)
        FP = np.sum(cfm[:, cls]) - TP
        FN = np.sum(cfm[cls, :]) - TP

        class_accuracy = (TP + TN) / (TP + TN + FP + FN)
        class_accuracies.append(class_accuracy)

    return class_accuracies


def calculate_specificity(y_true, y_pred, classes):
    cfm = metrics.confusion_matrix(y_true, y_pred, labels=classes)
    specificities = []
    for cls in classes:
        true_negatives = np.sum(cfm) - np.sum(cfm[cls, :]) - np.sum(cfm[:, cls]) + cfm[cls, cls]
        false_positives = np.sum(cfm[:, cls]) - cfm[cls, cls]

        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        specificities.append(specificity)

    return specificities


def calculate_class_metrics(y_true, y_pred, cls):
    accuracy = calculate_class_accuracy(y_true, y_pred, cls)
    precision = metrics.precision_score(y_true, y_pred, labels=cls, average=None)
    recall = metrics.recall_score(y_true, y_pred, labels=cls, average=None)
    specificity = calculate_specificity(y_true, y_pred, cls)
    f1 = metrics.f1_score(y_true, y_pred, labels=cls, average=None)
    return accuracy, precision, recall, specificity, f1


def calculate_class_weights(y_trues):
    class_counts = np.bincount(y_trues)
    total_counts = len(y_trues)
    class_weights = class_counts / total_counts
    return class_weights


def evaluate_model(task, model, test_loader):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []

    for image, label, weight in test_loader:

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

    if task == 'multi':
        classes = [i for i in range(3)]
    else:
        classes = [i for i in range(2)]
    
    auc = calculate_class_auc(y_trues, y_preds, classes)
    loss = np.round(np.mean(losses), 4)

    y_preds_classes = np.argmax(y_preds, axis=1)
    y_trues_classes = np.argmax(y_trues, axis=1)
    
    accuracy, precision, recall, specificity, f1 = calculate_class_metrics(y_trues_classes, y_preds_classes, classes)
    
    class_weights = calculate_class_weights(y_trues_classes)

    weighted_auc = np.average(auc, weights=class_weights)
    weighted_accuracy = np.average(accuracy, weights=class_weights)
    weighted_precision = np.average(precision, weights=class_weights)
    weighted_recall = np.average(recall, weights=class_weights)
    weighted_specificity = np.average(specificity, weights=class_weights)
    weighted_f1 = np.average(f1, weights=class_weights)

    with open(f'./result/{args.model}.csv', 'w') as writer:
        if task == 'binary':
            header = "auc,accuracy,precision,recall,specificity,f1"
            data_line = f"{auc[1]},{accuracy[1]},{precision[1]},{recall[1]},{specificity[1]},{f1[1]}"
        else:
            header = "cls,auc,accuracy,precision,recall,specificity,f1"
            data_lines = [f"{cls},{auc[cls]},{accuracy[cls]},{precision[cls]},{recall[cls]},{specificity[cls]},{f1[cls]}" for cls in classes]
            data_lines.append(f"weighted,{weighted_auc},{weighted_accuracy},{weighted_precision},{weighted_recall},{weighted_specificity},{weighted_f1}")
            data_line = "\n".join(data_lines)
        print(header, file=writer)
        print(data_line, file=writer)


def run(args):
    BASE_PATH = f"/media/new_partition/shoulder_MRI/npy_data/{args.plane}"
    LABEL_PATH = f"./label/id_grade.csv"

    id_grade = get_id_grade(LABEL_PATH)
    file_paths = np.array([os.path.join(BASE_PATH, file) for file in os.listdir(BASE_PATH)])
    labels = np.array(get_label(args.task, file_paths, id_grade))

    _, X_test, _, y_test = train_test_split(file_paths, labels, test_size=0.1, stratify=labels, random_state=42)
    test_dataset = MRDataset(args.task, X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)

    MODEL_BASE_PATH = "./models"
    mrnet = torch.load(f"{MODEL_BASE_PATH}/{args.model}")
    evaluate_model(args.task, mrnet, test_loader)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['binary', 'multi'])
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
