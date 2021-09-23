# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The goal of this project is to design a machine learning pipeline to identify credit card customers that are most likely to churn. Note that we make sure best coding practices are followed, from testing, logging, moduralizing and documenting. 


The data used in this project are available in [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code).

## Installation 
To make sure the project run smoothly make sure to install the run the following command

- `$pip install -r requirements.txt`

or just install the following: 

```
$pip install scikit-learn==0.22 shap pylint autopep8 && clear
```
## Running Files
The original python notebook `churn_notebook.ipynb` contains the code to be refactored.

Beside the notebook, there are mainly two important python files in this project. If one is interested 
in running the whole pipeline from testing and logging while simultaneously running 
the project, then run the following command

`$python churn_script_logging_and_tests.py` 

This file is organized in multiple test function to make it easier to detect an error
if there is one. 
- test_import
- test_eda
- test_encoder_helper
- test_perform_feature_engineering
- test_train_models

Thie python script will test each of the functions and provide any errors to a file stored in the `logs` folder.

Otherwise  the command below will only run the whole project (without testing) once.
The used data is stored in the Data folder. The remaining folders are explicit by 
their naming. 

`$python churn_library.py`

Running either one of the two python churn python files will generate trained model, that is 
saved in the model folder. Various figures for exploratory data analysis are generated
in `"./images/eda"`. Different metrics are also saved in `"./images/results/"` for both the Random Forrest classifier
together with the logistic regression classifier

You can also check the pylint score, as well as perform the auto-formatting using the following commands:

```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

The files here were formated using:
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```


