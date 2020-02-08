# Background
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, we have to employ different techniques to train and evaluate models with unbalanced classes. We will use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Final task is to evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

## Objectives
The goals of this challenge is to: 
- Implement machine learning models.
- Use resampling to attempt to address class imbalance.
- Evaluate the performance of machine learning models.

### Code File #1 (credit_risk_resampling.ipynb):

We use the imbalanced-learn library to resample the data and build and evaluate logistic regression classifiers using the resampled data. 

- First we Oversample the data using the RandomOverSampler and SMOTE algorithms
- Then we Undersample the data using the cluster centroids algorithm
- Finally we use a combination approach with the SMOTEENN algorithm.

For each of the above, we:
- Train a logistic regression classifier (from Scikit-learn) using the resampled data
- Calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics
- Generate a confusion_matrix
- Print the classification report (classification_report_imbalanced from imblearn.metrics)

### Lastly, we provide a brief summary and analysis of the models’ performance within the code file at the end.

In the summary we describe the precision and recall scores, as well as the balanced accuracy score. Additionally, include a final recommendation on the model to use, if any. 


### Code File #2 (credit_risk_ensemble.ipynb):

Here we train and compare two different ensemble classifiers (BalancedRandomForestClassifier and EasyEnsembleClassifier from imblearn.ensemble) to predict loan risk and evaluate each model. These modules combine resampling and model training into a single step. 

We have used 100 estimators for both classifiers, and completed the following steps for each model:
- Train the model and generate predictions
- Calculate the balanced accuracy score
- Generate a confusion matrix
- Print the classification report (classification_report_imbalanced from imblearn.metrics)
- For the BalancedRandomForestClassifier, print the feature importance, sorted in descending order (from most to least important feature), along with the feature score.


## Lastly, we have included a brief summary and analysis of the models’ performance within the code file. 
The summary describes the precision and recall scores, as well as the balanced accuracy score for each model. Additionally, it includes a final recommendation on the model to use, if any with a reasoning.


# NOTE: Analysis Summaries are included as a markdown within the code file, towards the end
