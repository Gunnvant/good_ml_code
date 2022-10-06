'''
Runs the training and eda functions for predicting customer churn
author: Gunnvant Singh
date: 5/10/22
'''

# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import constants
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def create_dependent_var(data_frame):
    '''
    creates churn variable

    input:
            pandas dataframe
    output:
            None
    '''
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)


def save_hist_plot(data_frame, col_name, plot_name):
    path_save = os.path.join(constants.PATH_IMG_EDA, plot_name)
    plt.figure(figsize=(20, 10))
    ax = data_frame[col_name].hist()
    ax.figure.savefig(path_save, dpi=300)


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''

    save_hist_plot(data_frame, col_name="Churn", plot_name='churn_hist.png')
    save_hist_plot(data_frame, col_name="Customer_Age", plot_name='cust_age_hist.png')

    plt.figure(figsize=(20, 10))
    path_save_bar = os.path.join(constants.PATH_IMG_EDA, 'marital_status.png')
    ax = data_frame['Marital_Status'].value_counts('normalize').plot(kind='bar')
    ax.figure.savefig(path_save_bar, dpi=300)

    path_save_histplot = os.path.join(
        constants.PATH_IMG_EDA, 'total_trans.png')
    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(path_save_histplot)

    path_save_heatmap = os.path.join(constants.PATH_IMG_EDA, 'heatmap.png')
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(path_save_heatmap)


def make_encoded_col(data_frame, col_name):
    '''
    creates encodings by groups, it computes the proportion of churn
    for each category

    input:
            data_frame: pandas dataframe
            col_name: categorical column for which encodings have to be computed
    output:
            data_frame: pandas dataframe with new column
    '''
    mapping = data_frame.groupby(col_name)['Churn'].mean().to_dict()
    encoded_col_name = col_name + "_Churn"
    data_frame[encoded_col_name] = data_frame[col_name].map(mapping)


def encoder_helper(data_frame, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for col in category_lst:
        make_encoded_col(data_frame, col)
    return data_frame


def perform_feature_engineering(data_frame, response='Churn'):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = data_frame[response]
    X = data_frame[constants.X_COLS]
    return train_test_split(
        X,
        y,
        test_size=constants.TEST_SIZE,
        random_state=constants.RANDOM_STATE)


def image_classification(y_train, y_test, pred_train, pred_test, model_name):
    '''
    creates and saves classification report for a model type and associated predictions
    input:
            y_train: training response values
            y_test:  test response values
            pred_train: training predictions
            pred_test: test predictions
            model_name: name of algorithm used
    output:
            None
    '''
    path_save = os.path.join(
        constants.PATH_IMG_RES,
        f'{model_name}_featimp.png')
    plt.figure(figsize=(20, 5))
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, pred_test)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, pred_train)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(path_save)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    image_classification(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        "logistic regression")
    image_classification(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        "random forest")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_lr(x_train, x_test, y_train):
    '''
    trains a logistic regression model
    inputs:
            x_train: X training data
            y_train: y training data
            x_test: X testing data
    outputs:
            pred_train: predictions on train data
            pred_test: predictions on test data
            model: final model instance
    '''
    clf = LogisticRegression(**constants.LOGISTIC_CONFIG)
    clf.fit(x_train, y_train)
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)
    path_model = os.path.join(constants.PATH_MODELS, 'logistic_model.pkl')
    joblib.dump(clf, path_model)
    return pred_train, pred_test, clf


def train_rf(x_train, x_test, y_train):
    '''
    trains a random forest model
    inputs:
            x_train: X training data
            y_train: y training data
            x_test: X testing data
    outputs:
            pred_train: predictions on train data
            pred_test: predictions on test data
            model: final model instance
    '''
    rfc = RandomForestClassifier(random_state=constants.RANDOM_STATE)
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=constants.PARAM_GRID,
        cv=constants.CV)
    cv_rfc.fit(x_train, y_train)
    pred_train = cv_rfc.best_estimator_.predict(x_train)
    pred_test = cv_rfc.best_estimator_.predict(x_test)
    path_model = os.path.join(constants.PATH_MODELS, 'rfc_model.pkl')
    joblib.dump(cv_rfc.best_estimator_, path_model)
    return pred_train, pred_test, cv_rfc


def save_roc_plots(lr, rf, x_test, y_test):
    '''
    saves roc plots for logitic regression and randomforest models
    inputs:
            lr: trained logistic regression model
            rf: trained random forest model
            x_test: X testing data
            y_test: y testing data
    '''
    path_plot = os.path.join(constants.PATH_IMG_RES, 'roc_plot.png')
    lrc_plot = plot_roc_curve(lr, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        rf.best_estimator_,
        x_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(path_plot)


def train_models(x_train, x_test, y_train, y_test, data_frame):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    y_train_preds_lr, y_test_preds_lr, lr = train_lr(x_train, x_test, y_train)
    y_train_preds_rf, y_test_preds_rf, rf = train_rf(x_train, x_test, y_train)
    feat_output_path = os.path.join(constants.PATH_IMG_RES, 'feature_imp.png')
    X_data = data_frame[constants.X_COLS]
    feature_importance_plot(rf, X_data, feat_output_path)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    save_roc_plots(lr, rf, x_test, y_test)


if __name__ == "__main__":
    path_data = constants.PATH_DATA
    data = import_data(path_data)
    create_dependent_var(data)
    perform_eda(data)
    data = encoder_helper(data, constants.CATEGORY_LIST)
    X_train, X_test, y_train, y_test = perform_feature_engineering(data)
    train_models(X_train, X_test, y_train, y_test, data)
