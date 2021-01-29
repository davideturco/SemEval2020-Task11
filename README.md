# SevEval2020-Task11
This repository contains the code for SemEval 2020 - Task 11 "Detection of propaganda techniques in news article". The proposed solution makes use of HMM, Bi-LSTM + CRF and RoBERTa base model.

## Repository structure
The main parts of the program can be found in directories ``SI/`` and ``TC/`` that contain the code for the Span Identification and Technique Classification tasks respectively. They share the same structure:
- `data_processing.py`: useful functions for processing text
- `hmm_train.py`: script for training and evaluating HMM
- `flair_train.py`: script for training deep learning models 
- `predict.py`: script for making predictions and writing them in submission format

## Requirements and set up
Create a new Conda environment with the specified packages installed: ```conda create --name <env> --file <environment.yml>```<br>
Activate the environment ```conda activate <env>```<br>
Run the desired script
