import os
import torch
import numpy as np
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dataloader import MRDataset, get_id_grade, get_label
from eval import calculate_class_metrics, calculate_class_weights


class FusionModel:
    def __init__(self, models, task):
        """
        Initialize the fusion model.
        
        :param models: Dictionary of pretrained models for each plane {'Axial': model_axial, 'Coronal': model_coronal, 'Sagittal': model_sagittal}.
        :param task_type: Type of task, either 'Binary' or 'Multi'.
        """
        self.models = models
        self.task_type = task
        self.lr_model = LogisticRegression()

        self.train_dataset = {'features': [], 'labels': []}
        self.test_dataset = {'features': [], 'labels': []}

        self.preditions = None

    def prepare_dataset(self, dataset_loaders, train=True):
        """
        Prepare the dataset for the Logistic Regression model.

        :param dataset_loaders: Dictionary of data loaders for each plane.
        :return: Prepared features and labels for the LR model.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features = []
        labels = []

        axial_loader = dataset_loaders['axial']
        coronal_loader = dataset_loaders['coronal']
        sagittal_loader = dataset_loaders['sagittal']

        for i, ((axial_data, axial_label, _), (coronal_data, coronal_label, _), (sagittal_data, sagittal_label, _)) in enumerate(zip(axial_loader, coronal_loader, sagittal_loader)):
            print(f"Building Datasets: {i}/{len(axial_loader)}")
            # Process and predict for each plane
            axial_pred = torch.sigmoid(self.models['axial'](axial_data.to(device))).detach().cpu().numpy().flatten()
            coronal_pred = torch.sigmoid(self.models['coronal'](coronal_data.to(device))).detach().cpu().numpy().flatten()
            sagittal_pred = torch.sigmoid(self.models['sagittal'](sagittal_data.to(device))).detach().cpu().numpy().flatten()
            
            # Combine predictions from all planes
            combined_pred = np.concatenate((axial_pred, coronal_pred, sagittal_pred))
            features.append(combined_pred)

            # Assume labels are the same for all planes for a single data point
            class_label = np.argmax(axial_label[0].numpy(), axis=1)
            labels.append(class_label[0])
        
        if train:
            self.train_dataset['features'] = np.array(features)
        else:
            self.test_dataset['features'] = np.array(features)
    
        if train:
            self.train_dataset['labels'] = np.array(labels)
        else:
            self.test_dataset['labels'] = np.array(labels)

    def fit(self):
        """
        Fit the Logistic Regression model using the predictions from the individual models.

        :param dataset: Dataset containing the data for fitting the models.
        """
        X, y = self.train_dataset['features'], self.train_dataset['labels']
        self.lr_model.fit(X, y)


    def predict(self):
        """
        Predict the output for a batch of data using the fusion model.

        :param dataset_loaders: Dictionary of data loaders to predict the output for.
        :return: List of predictions for the dataset.
        """
        X = self.test_dataset['features']
        self.preditions = self.lr_model.predict(X)
    
    def evaluate(self, k):
        if self.task_type == 'multi':
            classes = [i for i in range(3)]
        else:
            classes = [i for i in range(2)]
    
        y_trues = self.test_dataset['labels']
        y_preds = self.preditions
        accuracy, precision, recall, specificity, f1 = calculate_class_metrics(y_trues, y_preds, classes)
        
        class_weights = calculate_class_weights(y_trues)

        weighted_accuracy = np.average(accuracy, weights=class_weights)
        weighted_precision = np.average(precision, weights=class_weights)
        weighted_recall = np.average(recall, weights=class_weights)
        weighted_specificity = np.average(specificity, weights=class_weights)
        weighted_f1 = np.average(f1, weights=class_weights)

        with open(f'./result/fusion-{self.task_type}-{k}.csv', 'w') as writer:
            if self.task_type == 'binary':
                header = "accuracy,precision,recall,specificity,f1"
                data_line = f"{accuracy[1]},{precision[1]},{recall[1]},{specificity[1]},{f1[1]}"
            else:
                header = "cls,accuracy,precision,recall,specificity,f1"
                data_lines = [f"{cls},{accuracy[cls]},{precision[cls]},{recall[cls]},{specificity[cls]},{f1[cls]}" for cls in classes]
                data_lines.append(f"weighted,{weighted_accuracy},{weighted_precision},{weighted_recall},{weighted_specificity},{weighted_f1}")
                data_line = "\n".join(data_lines)
            print(header, file=writer)
            print(data_line, file=writer)


def run(args):
    LABEL_PATH = f"./label/id_grade.csv"
    id_grade = get_id_grade(LABEL_PATH)

    BASE_PATH = "/media/new_partition/shoulder_MRI/npy_data"

    file_lists = []
    for plane in ('axial', 'coronal', 'sagittal'):
        path = os.path.join(BASE_PATH, plane)
        file_lists.append(set(os.listdir(path)))

    fusion_files = sorted(set.intersection(*file_lists))

    train_datasets = {}
    test_datasets = {}
    for plane in ('axial', 'coronal', 'sagittal'):
        file_paths = np.array([os.path.join(BASE_PATH, plane, file) for file in fusion_files])
        labels = np.array(get_label(args.task, file_paths, id_grade))

        X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, stratify=labels, random_state=42)
        train_dataset = MRDataset(args.task, X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=11, drop_last=False)
        
        test_dataset = MRDataset(args.task, X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=-False, num_workers=11, drop_last=False)

        train_datasets[plane] = train_loader
        test_datasets[plane] = test_loader

    model_axial = torch.load(f"./models/{args.axial_model}")
    model_coronal = torch.load(f"./models/{args.coronal_model}")
    model_sagittal = torch.load(f"./models/{args.sagittal_model}")

    if torch.cuda.is_available():
        model_axial.cuda()
        model_coronal.cuda()
        model_sagittal.cuda()
    
    models = {'axial': model_axial, 'coronal': model_coronal, 'sagittal': model_sagittal}
    fusion_model = FusionModel(models, args.task)
    fusion_model.prepare_dataset(train_datasets, train=True)
    fusion_model.prepare_dataset(test_datasets, train=False)
    fusion_model.fit()
    fusion_model.predict()
    fusion_model.evaluate(args.k_fold)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['binary', 'multi'])
    parser.add_argument('-a', '--axial_model', type=str, required=True)
    parser.add_argument('-c', '--coronal_model', type=str, required=True)
    parser.add_argument('-s', '--sagittal_model', type=str, required=True)
    parser.add_argument('-k', '--k_fold', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
