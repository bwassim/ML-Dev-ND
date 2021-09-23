"""
This file operates testing and logging for the churn library file

Author: Ouassim bara
Data: 22 Sep 2021

"""
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s"
)


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err
    return df

def test_eda(perform_eda, df):

    """
    test perform eda function
    """

    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    perform_eda(df)


PATH = "./images/eda"

try:
    assert len(os.listdir(PATH)) > 0
    logging.info("Testing perform_eda: SUCCESS")
except AssertionError as err:
    logging.error("Testing perform_eda: There is no saved image")
    raise err


def test_encoder_helper(encoder_helper, df):
    """
    test encoder helper
    """
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"
    ]

    df = encoder_helper(df, cat_columns, 'Churn')

    try:
        for col in cat_columns:
            assert col in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe appears to be missing the "
            "transformed categorical columns")
        return err

    return df


def test_perform_feature_engineering(df, perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    # make sure
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The train and test data are not returned"
        )
        raise err
    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    """
    test train_models
    """
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(
        X_train, X_test, y_train, y_test
    )
    try:
        assert len(y_train_preds_rf) > 0
        assert len(y_test_preds_rf) > 0
        assert len(y_train_preds_lr) > 0
        assert len(y_test_preds_lr) > 0
        logging.info("Test train models: ")
    except AssertionError as err:
        logging.error(
            "Test train models: Failed to predict results from training and test data"
        )
    # check saved files
    path = "./images/results"
    try:
        assert len(os.listdir(path)) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Prediction figures not found")
        raise err


if __name__ == "__main__":
    DATA_FRAME = test_import(cls.import_data)
    DATA_FRAME.head()
    test_eda(cls.perform_eda, DATA_FRAME)
    DATA_FRAME = test_encoder_helper(cls.encoder_helper, DATA_FRAME)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        DATA_FRAME, cls.perform_feature_engineering
    )
    test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
