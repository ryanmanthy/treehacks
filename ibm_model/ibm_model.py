# Create scikit xgboost model for ibm medical expenditure dataset
# Outputs prediction results to csv file

import sys
sys.path.insert(0, '../')
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# LIME
from aif360.datasets.lime_encoder import LimeEncoder
import lime
from lime.lime_tabular import LimeTabularExplainer

# XGBoost
import xgboost as xgb

np.random.seed(1)



# Load dataset name
def load_dataset(dataset_name):
    if dataset_name == 'meps19':
        dataset = MEPSDataset19()
    elif dataset_name == 'meps20':
        dataset = MEPSDataset20()
    elif dataset_name == 'meps21':
        dataset = MEPSDataset21()
    else:
        raise ValueError('Dataset not found.')
    return dataset

# Create train, val, test datasets
def create_data(dataset_name='meps19'):
    dataset = load_dataset(dataset_name)
    (train,val,test) = dataset.split([0.5, 0.8], shuffle=True)

    sens_ind = 0
    sens_attr = train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                        train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                        train.privileged_protected_attributes[sens_ind]]

    return {'train': train, 'val': val, 'test': test, 
            'unprivileged_groups': unprivileged_groups, 
            'privileged_groups': privileged_groups}


# Describes dataset
def describe(train=None, val=None, test=None):
    if train is not None:
        print("Training Dataset shape")
        print(train.features.shape)
    if val is not None:
        print("Validation Dataset shape")
        print(val.features.shape)
    print("Test Dataset shape")
    print(test.features.shape)
    print("Favorable and unfavorable labels")
    print(test.favorable_label, test.unfavorable_label)
    print("Protected attribute names")
    print(test.protected_attribute_names)
    print("Privileged and unprivileged protected attribute values")
    print(test.privileged_protected_attributes, 
          test.unprivileged_protected_attributes)
    print("Dataset feature names")
    print(test.feature_names)


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

def predict_model(dataset, model, thresh=0.5):
    try:
        # sklearn classifier
        y_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    y_pred = (y_pred_prob[:, pos_ind] > thresh).astype(np.float64)
    return (y_pred_prob, y_pred)

def test_model(dataset, model, thresh_arr, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['thresh'].append(thresh)
        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    
    return metric_arrs


if __name__ == "__main__":
    # Load dataset
    data = create_data(dataset_name='meps19')
    train = data['train']
    val = data['val']
    test = data['test']

    # Save datasets to csv
    train.convert_to_dataframe()[0].to_csv('meps19_train.csv', index=False)
    val.convert_to_dataframe()[0].to_csv('meps19_val.csv', index=False)
    test.convert_to_dataframe()[0].to_csv('meps19_test.csv', index=False)

    unprivileged_groups = data['unprivileged_groups']
    privileged_groups = data['privileged_groups']

    # Describe dataset
    describe(train=train, val=val, test=test)

    # Train model
    model = train_model(train=train, model_name='rf')

    # Test model
    thresh_arr = np.linspace(0.01, 0.5, 50)
    metric_arrs = test_model(test, model, thresh_arr, unprivileged_groups, privileged_groups)

    # Save metrics to csv
    metric_results = pd.DataFrame(metric_arrs)
    lr_orig_best_ind = np.argmax(metric_results['bal_acc'])
    final_metric = metric_results.iloc[[lr_orig_best_ind]]
    final_metric.to_csv('metrics.csv', index=False)


    # Save predictions to csv
    y_pred_train = predict_model(train, model, thresh=thresh_arr[lr_orig_best_ind])
    y_pred_val = predict_model(val, model, thresh=thresh_arr[lr_orig_best_ind])
    y_pred_test = predict_model(test, model, thresh=thresh_arr[lr_orig_best_ind])

    train_df = pd.DataFrame({'y_pred': y_pred_train[1], # model predictions
                             'y_pred_prob': y_pred_train[0][:, 1], # model prediction probabilities
                             'y_true': train.labels.ravel(), # true labels
                             'instance_weights': train.instance_weights, # instance weights
                             'protected_attribute_0': train.protected_attributes[:, 0], # sensitive features
                             })
    train_df.to_csv('train_predictions.csv', index=False)

    val_df = pd.DataFrame({'y_pred': y_pred_val[1], # model predictions
                            'y_pred_prob': y_pred_val[0][:, 1], # model prediction probabilities
                            'y_true': val.labels.ravel(), # true labels
                            'instance_weights': val.instance_weights, # instance weights
                            'protected_attribute_0': val.protected_attributes[:, 0], # sensitive features
                            })
    val_df.to_csv('val_predictions.csv', index=False)

    test_df = pd.DataFrame({'y_pred': y_pred_test[1], # model predictions
                            'y_pred_prob': y_pred_test[0][:, 1], # model prediction probabilities
                            'y_true': test.labels.ravel(), # true labels
                            'instance_weights': test.instance_weights, # instance weights
                            'protected_attribute_0': test.protected_attributes[:, 0], # sensitive features
                            })
    test_df.to_csv('test_predictions.csv', index=False)

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)







