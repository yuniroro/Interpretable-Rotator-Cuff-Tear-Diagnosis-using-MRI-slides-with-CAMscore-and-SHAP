# Interpretable-Rotator-Cuff-Tear-Diagnosis-using-MRI-slides-with-CAMscore-and-SHAP

![](./images/RCTs.jpg)

## Overview

This repository contains the code and parameters for the project presented at SPIE Medical Imaging 2024, titled "Interpretable Rotator Cuff Tear Diagnosis using MRI slides with CAMscore and SHAP." The project focuses on developing a Computer-Aided Diagnosis (CAD) model for age-related musculoskeletal disorders, specifically Rotator Cuff Tears (RCTs), in the shoulder region. The goal is to enhance the interpretability of the CAD model using three-plane MRI slides and diagnostic outcomes.

## Key Features

- **CAMscore:** Introduces a CAMscore mechanism to identify the importance of individual MRI slides in the decision-making process.
- **SHAP (SHapley Additive exPlanations):** Utilizes SHAP values for interpreting the model's output and understanding feature importance.

![](./images/project_overview.jpg?scale=0.7)

## Dataset
The dataset used in this research is derived from the repository available at [MRI-based Diagnosis of Rotator Cuff Tears using Deep Learning and Weighted Linear Combinations](https://github.com/powersimmani/MRI-based-Diagnosis-of-Rotator-Cuff-Tears-using-Deep-Learning-and-Weighted-Linear-Combinations). This dataset includes MRI slides for diagnosing Rotator Cuff Tears.


## File Structure
```
project
├── data 
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

### Best Models
[Models Download](https://doi.org/10.6084/m9.figshare.25035767.v1)

### CAMscore
![](./images/GradCAM.jpg?scale=0.7)

### SHAP
![](./images/SHAP.jpg?scale=0.7)


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




## References
- Besbes, A., “MRNet: GitHub Repository,” (2023). [GitHub Repository](https://github.com/ahmedbesbes/mrnet).
- Bien, N., Rajpurkar, P., Ball, R. L., Irvin, J., Park, A., Jones, E., Bereket, M., Patel, B. N., Yeom, K. W., Shpanskaya, K., Halabi, S., Zucker, E., Fanton, G., Amanatullah, D. F., Beaulieu, C. F., Riley, G. M., Stewart, R. J., Blankenberg, F. G., Larson, D. B., Jones, R. H., Langlotz, C. P., Ng, A. Y., and Lungren, M. P., “Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet,” PLOS Medicine 15, 1–19 (11 2018).
- Kim, M., Park, H.-m., Kim, J. Y., Kim, S. H., Hoeke, S., and De Neve, W., “MRI-based Diagnosis of Rotator Cuff Tears using Deep Learning and Weighted Linear Combinations,” in Proceedings of the 5th Machine Learning for Healthcare Conference, Doshi-Velez, F., Fackler, J., Jung, K., Kale, D., Ranganath, R., Wallace, B., and Wiens, J., eds., Proceedings of Machine Learning Research 126, 292–308, PMLR (07–08 Aug 2020).
