# Interpretable-Rotator-Cuff-Tear-Diagnosis-using-MRI-slides-with-CAMscore-and-SHAP

## Project Overview
This research focuses on developing a Computer-Aided Diagnosis (CAD) model for age-related musculoskeletal disorders, specifically Rotator Cuff Tears (RCTs), that occur in the shoulder region. The project involves using three-plane MRI slides coupled with diagnostic outcomes to enhance the interpretability of the CAD model. The MRNet architecture is utilized, with training on each anatomical plane and a fusion of results through logistic regression.

## File Structure
project  
├── label  
├── models  
├── dataloader.py  
├── eval.py  
├── fusion.py  
├── model.py  
└── train.py  

train.py: Script to train the model on MRI data.  
dataloader.py: Handles loading and preprocessing of MRI data.  
eval.py: Evaluates the performance of the model.  
fusion.py: Responsible for fusing results from different planes using logistic regression.  
model.py: Defines the MRNet architecture used in this research.  

## Experimental Results and Analysis
Discusses the outcomes of the CAD model, highlighting the F1-score of 0.9508 achieved when fusing three planes.
Comparison of diagnostic efficacy across sagittal, axial, and coronal planes.
Introduction of CAMscore, a method using GradCAM, to quantitatively assess the diagnostic relevance of individual MRI slides.
