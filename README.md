# Background
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, we have to employ different techniques to train and evaluate models with unbalanced classes. We will use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Final task is to evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

## Objectives
The goals of this challenge is to: 
- Implement machine learning models.
- Use resampling to attempt to address class imbalance.
- Evaluate the performance of machine learning models.

Code File #1 (credit_risk_resampling.ipynb):
We use the imbalanced-learn library to resample the data and build and evaluate logistic regression classifiers using the resampled data. 

- First we Oversample the data using the RandomOverSampler and SMOTE algorithms
- Then we Undersample the data using the cluster centroids algorithm
- Finally we use a combination approach with the SMOTEENN algorithm.

For each of the above, we:
- Train a logistic regression classifier (from Scikit-learn) using the resampled data
- Calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics
- Generate a confusion_matrix
- Print the classification report (classification_report_imbalanced from imblearn.metrics)

AND Lastly, we provide a brief summary and analysis of the models’ performance towards the end of the code file.

In the summary we describe the precision and recall scores, as well as the balanced accuracy score. Additionally, include a final recommendation on the model to use, if any. 

