"""
library doc string
This script highlights two approaches for classification.
One based on Random Forrest and the other a simple logistic
regression model. # Comparison are displayed.

Author: Ouassim Bara
Date: Sep 21 2021

"""
# import libraries
# import os
# os.environ['QT_QPA_PLATFORM']='offscreen'

# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
# import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    data_frame = pd.read_csv(pth)
    data_frame.head()
    return data_frame


def perform_eda(data_frame):
    """
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    """
    data_frame["Churn"] = data_frame["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    plt.figure(figsize=(20, 10))
    data_frame["Churn"].hist()
    plt.savefig("./images/eda/churn_distribution.png")

    plt.figure(figsize=(20, 10))
    data_frame["Customer_Age"].hist()
    plt.savefig("./images/eda/customer_age_distribution.png")

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig("./images/eda/marital_status_distribution.png")

    plt.figure(figsize=(20, 10))
    sns.displot(data_frame["Total_Trans_Ct"])
    plt.savefig("./images/eda/total_transaction_distribution.png")

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="YlGnBu", linewidths=2)
    plt.savefig("./images/eda/heatmap.png")


def encoder_helper(data_frame, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    """
    for category in category_lst:
        category_lst = []
        category_groups = data_frame.groupby(category).mean()[response]
        for val in data_frame[category]:
            category_lst.append(category_groups.loc[val])
        data_frame[category + "_" + response] = category_lst

    return data_frame


def perform_feature_engineering(data_frame, response):
    """
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    data_frame = encoder_helper(data_frame, cat_columns, "Churn")

    y = data_frame["Churn"]

    X = pd.DataFrame()
    X[keep_cols] = data_frame[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(X_train)

    return (X_train, X_test, y_train, y_test)


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
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
    """
    plt.figure()
    plt.rc("figure", figsize=(8, 8))
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Random Forest Test (below) Random Forest Train (above)"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("./images/results/rf_results.png")
    plt.close()

    plt.figure()
    plt.rc("figure", figsize=(8, 8))
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Logistic Regression Test (below) Logistic Regression Train (above)"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("./images/results/logistic_results.png")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # calculate feature importance
    importances = model.feature_importances_

    # sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feauture names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, bbox_inches="tight")
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }
    # define the grid search
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # training random forrest and logistic regression
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # prediction for random forrest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # prediction for logistic regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save the scores
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    # Save roc curve
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("./images/results/roc_curve_result.png")
    plt.close()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")

    # Save feature importance plot
    feature_importance_plot(
        cv_rfc.best_estimator_, X_train, "./images/results/feature_importances.png"
    )

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


if __name__ == "__main__":

    df = import_data(r"./data/bank_data.csv")

    perform_eda(df)

    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")

    train_models(X_train, X_test, y_train, y_test)
