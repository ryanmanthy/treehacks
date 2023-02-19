# Measure biases and disparities in model predictions using fairness metrics
import sys
sys.path.insert(0, '../')
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
#from aif360.explainers import MetricTextExplainer

# Dataset
from aif360.datasets import BinaryLabelDataset



'''
Function: measure_disparity

dataset: Pandas dataframe containing columns:
    - y_pred: Model predictions
    - y_pred_prob: Model prediction probabilities
    - y_true: Ground truth labels 
    - instance_weights: Weights for each instance
    - protected_attribute_{idx}: Sensitive features
        - Each column represents a different sensitive feature

unprivileged_groups: List of dictionaries containing the unprivileged groups
    - Each dictionary contains the values for each sensitive feature
    - If None, assumes binary labels where sensitive feature is 0

privileged_groups: List of dictionaries containing the privileged groups
    - Each dictionary contains the values for each sensitive feature
    - If None, assumes binary labels where sensitive feature is 1

Returns: One value per protected class measuring discrimination for each metric used
'''
def run_measure_disparity(predictions_csv, unpriviledged_groups=None, priviledged_groups=None):
    predictions = pd.read_csv(predictions_csv)
    return measure_disparity(predictions, unpriviledged_groups, priviledged_groups)

def measure_disparity(predictions, unprivileged_groups=None, privileged_groups=None):
    protected_attribute_names = [col for col in predictions.columns if 'protected_attribute' in col]

    # Create dataset
    truth_dataset = BinaryLabelDataset(df=predictions, label_names=['y_true'], 
                                 protected_attribute_names=protected_attribute_names, 
                                 favorable_label=1, unfavorable_label=0)

    pred_dataset = truth_dataset.copy()
    pred_dataset.labels = predictions['y_pred'].values

    if unprivileged_groups is None:
        unprivileged_groups = [{col: 0 for col in protected_attribute_names}]

    if privileged_groups is None:
        privileged_groups = [{col: 1 for col in protected_attribute_names}]

    fairness_metrics = dict()
    metric = ClassificationMetric(
                truth_dataset, pred_dataset,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

    fairness_metrics['bal_acc'] = (metric.true_positive_rate()
                                    + metric.true_negative_rate()) / 2
    fairness_metrics['avg_odds_diff'] = metric.average_odds_difference()
    fairness_metrics['disp_imp'] = metric.disparate_impact()
    fairness_metrics['stat_par_diff'] = metric.statistical_parity_difference()
    fairness_metrics['eq_opp_diff'] = metric.equal_opportunity_difference()
    fairness_metrics['theil_ind'] = metric.theil_index()
    
    return fairness_metrics

if __name__ == "__main__":
    # Load dataset
    train_predictions = pd.read_csv("ibm_model/train_predictions.csv")
    val_predictions = pd.read_csv("ibm_model/val_predictions.csv")

    # Measure disparity
    train_disparity = measure_disparity(train_predictions)
    val_disparity = measure_disparity(val_predictions)

    print()
    print("Training set disparity:")
    print(train_disparity)
    print()
    print("Validation set disparity:")
    print(val_disparity)
