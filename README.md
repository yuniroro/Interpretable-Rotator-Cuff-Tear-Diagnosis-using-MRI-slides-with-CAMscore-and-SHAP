# Interpretable-Rotator-Cuff-Tear-Diagnosis-using-MRI-slides-with-CAMscore-and-SHAP

## Project Overview
This research focuses on developing a Computer-Aided Diagnosis (CAD) model for age-related musculoskeletal disorders, specifically Rotator Cuff Tears (RCTs), that occur in the shoulder region. The project involves using three-plane MRI slides coupled with diagnostic outcomes to enhance the interpretability of the CAD model. The MRNet architecture is utilized, with training on each anatomical plane and a fusion of results through logistic regression.  

## File Structure
```
project  
├── label  
├── models  
├── dataloader.py  
├── eval.py  
├── fusion.py  
├── model.py  
└── train.py  
```  

- train.py: Script to train the model on MRI data.  
- dataloader.py: Handles loading and preprocessing of MRI data.  
- eval.py: Evaluates the performance of the model.  
- fusion.py: Responsible for fusing results from different planes using logistic regression.  
- model.py: Defines the MRNet architecture used in this research.    

## Experimental Results and Analysis
Discusses the outcomes of the CAD model, highlighting the F1-score of 0.9508 achieved when fusing three planes.
Comparison of diagnostic efficacy across sagittal, axial, and coronal planes.
Introduction of CAMscore, a method using GradCAM, to quantitatively assess the diagnostic relevance of individual MRI slides.  

## Usage
### Train
To train the model, use the `train.py` script with various command-line arguments to specify the training parameters. For example:

```
python train.py --plane [sagittal/coronal/axial] --task [binary/multi] --model MRNet --cuda 0 --seed 42 --augment 1 --lr_scheduler plateau --gamma 0.5 --epochs 50 --lr 1e-5 --save_model 1 --patience 5 --log_every 100
```

Options:  
- `-p`, `--plane`: Specify the MRI plane (sagittal, coronal, or axial).
- `-t`, `--task`: Choose between binary or multi-class tasks.
- `-m`, `--model`: Model to use, default is 'MRNet'.
- `--cuda`: CUDA device number.
- `--seed`: Random seed for reproducibility.
- `--augment`: Enable or disable data augmentation.
- `--lr_scheduler`: Type of learning rate scheduler.
- `--gamma`: Learning rate decay factor.
- `--epochs`: Number of training epochs.
- `--lr`: Learning rate.
- `--save_model`: Save the model post-training.
- `--patience`: Patience for early stopping.
- `--log_every`: Frequency of logging training progress.

### Evaluation
For model evaluation, use the `eval.py` script. Provide the plane, task type, and model name for evaluation:

```
python eval.py --plane [sagittal/coronal/axial] --task [binary/multi] --model [ModelName]
```

Options:  
- `-p`, `--plane`: MRI plane for evaluation.
- `-t`, `--task`: Type of task for evaluation.
- `-m`, `--model`: Model used for evaluation.



