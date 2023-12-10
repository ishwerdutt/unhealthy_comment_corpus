# Fine-tuning Roberta for Unhealthy Comment Classification

This repository contains the code used to fine-tune a pre-trained RoBERTa-base model on a dataset for classifying unhealthy comments. The notebook `Fine_tuning_Roberta-unhealthy Comment Corpus.ipynb` details the process followed for fine-tuning the model.

## Overview

The goal of this project was to create a machine learning model capable of identifying and classifying unhealthy comments. The model is based on the RoBERTa transformer architecture, a robust and effective language model. By fine-tuning this model on a specific dataset containing unhealthy comment examples, we aimed to achieve better classification accuracy and performance for identifying such comments.

## Contents

- `Fine_tuning_Roberta-unhealthy Comment Corpus.ipynb`: This Jupyter Notebook contains the code used for fine-tuning the RoBERTa model. It covers data preprocessing, model configuration, training, evaluation, and model saving.
- `requirements.txt`: Lists the necessary Python packages and their versions required to run the notebook.

## Usage

To replicate the fine-tuning process or use the fine-tuned model for classifying unhealthy comments, follow these steps:

1. **Environment Setup**:
   - Install the required dependencies listed in `requirements.txt` using `pip install -r requirements.txt`.
   - Ensure Python 3.x is installed on your system.

2. **Data Preparation**:
   - Obtain the dataset containing unhealthy comments. Ensure it is formatted appropriately for model training and evaluation.

3. **Fine-tuning the Model**:
   - Open and run the `Fine_tuning_Roberta-unhealthy Comment Corpus.ipynb` notebook in a Jupyter environment.
   - Adjust hyperparameters, if necessary, for your specific dataset and computational resources.

4. **Evaluation and Inference**:
   - Evaluate the fine-tuned model's performance using relevant metrics (accuracy, precision, recall, etc.) on a separate test/validation set.
   - Use the saved model for inference on new comments to classify them as healthy or unhealthy.

## Model Performance

The performance of the fine-tuned RoBERTa model can be assessed using various evaluation metrics, such as accuracy, precision, recall, and F1-score, on a held-out test set. Details of the model's performance on the dataset used for fine-tuning will be available within the notebook.


