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

from aif360.preprocessing import Reweighing


# Scalers
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# XGBoost
import xgboost as xgb

np.random.seed(1)


"""
“mitigate_disparity.py” takes in a model development dataset (training and test datasets) 
that your algorithm has not seen before and generates a new, optimally fair/debiased model 
that can be used to make new predictions.

1. Inputs
(1) A model development dataset that contains information on:
(i)    Model features
(ii)   Model label
(iii)  Sample weights
(iv)  Demographic data on protected and reference classes

2. Outputs
(1)  The fair/debiased model object, taking the form of a sklearn-style python object with the following functions accessible:
(i)    .fit() – trains the model
(ii)   .predict() / .predict_proba() – makes predictions using new data
(iii)  .transform() – filters or modifies input data, if applicable
(2)  [Optional] graphics/visualization, useful formatted output
"""
def mitigate_disparity(dataset, unprivileged_groups=None, privileged_groups=None, model=None, model_name='rf'):
    if (type(dataset) == BinaryLabelDataset):
        sens_ind = 0
        sens_attr = dataset.protected_attribute_names[sens_ind]

        if unprivileged_groups is None:
            unprivileged_groups = [{sens_attr: v} for v in
                    dataset.unprivileged_protected_attributes[sens_ind]]
        if privileged_groups is None:
            privileged_groups = [{sens_attr: v} for v in
                    dataset.privileged_protected_attributes[sens_ind]]

    if model is None:
        # Reweights based on protected class
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
        dataset = RW.fit_transform(dataset)
        fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
        model = train_model(train=dataset, model_name=model_name, fit_params=fit_params)

    debiased_model = model
    return debiased_model


# Train model
def train_model(train=None, model_name='rf', model_params=None, fit_params=dict()):
    if model_name == 'rf':
        model = RandomForestClassifier(n_estimators=500, min_samples_leaf=25)
        if len(fit_params) == 0:
            fit_params = {'randomforestclassifier__sample_weight': train.instance_weights}
    elif model_name == 'lr':
        model = LogisticRegression()
    elif model_name == 'xgb':
        model = xgb.XGBClassifier()
    else:
        raise ValueError('Model not found.')

    model = make_pipeline(StandardScaler(), model)

    model.fit(train.features, train.labels.ravel(), **fit_params)
    return model