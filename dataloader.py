import torch
import torch.utils.data as data
import numpy as np


def get_id_grade(LABEL_PATH):
    reader = open(LABEL_PATH, 'r')
    id_grade = {}
    for line in reader:
        sub_id, grade = line.strip().split(',')
        id_grade[sub_id] = grade
    return id_grade


def get_label(task, file_paths, id_grade):
    labels = []
    for file_path in file_paths:
        sub_id = file_path.split('/')[-1].replace('.npy', '')
        grade = int(id_grade[sub_id])
        if task == 'binary':
            if grade <= 1:
                labels.append(0)
            else:
                labels.append(1)
        else:
            if grade <= 1:
                labels.append(0)
            elif grade == 2:
                labels.append(1)
            else:
                labels.append(2)
    return labels


class MRDataset(data.Dataset):
    def __init__(self, task, paths, labels, transform=None, weights=None):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.task = task

        self.transform = transform
        if weights is None:
            if task == 'multi':
                class_counts = np.array([np.sum(labels == i) for i in range(3)])
                weights = len(labels) / class_counts
                self.weights = torch.FloatTensor(weights)
            else:
                pos = np.sum(self.labels)
                neg = len(self.labels) - pos
                self.weights = torch.FloatTensor([1, neg / pos])
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label_idx = self.labels[index]
        
        if self.task == 'binary':
            label = torch.FloatTensor([[0, 0]])
            label[0][label_idx] = 1
        else:
            label = torch.FloatTensor([[0, 0, 0]])
            label[0][label_idx] = 1

        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        return array, label, self.weights
