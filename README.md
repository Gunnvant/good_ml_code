# Predict Customer Churn
This contains the code for predicting customer churn for the customers of a bank. The initial prototype model is trained in a jupyter notebook named `churn_notebook.ipynb`. From this notebook the code has been refactored using best coding practices. Details of that are in the next section.

## Project Description
In order for the ML code to be in a deployable state, the first requirement is to put the prototype code into a form which:
1. Is easily readable: This is achieved by following pep8 guidelines
2. Is optimized: This is done by changing sub-optimal code constructs into more optimal ones, eg using pandas or numpy operations rather than running for loops
3. Can be tested: This is done using unit tests and logging
4. Is modular: This is done by breaking code into functions and modules

## Files and data description
The project has the following structure:

📦customer_churn
 ┣ 📂data
 ┃ ┗ 📜bank_data.csv
 ┣ 📂images
 ┃ ┣ 📂eda
 ┃ ┃ ┣ 📜churn_hist.png
 ┃ ┃ ┣ 📜cust_age_hist.png
 ┃ ┃ ┣ 📜heatmap.png
 ┃ ┃ ┣ 📜marital_status.png
 ┃ ┃ ┗ 📜total_trans.png
 ┃ ┣ 📂results
 ┃ ┃ ┣ 📜feature_imp.png
 ┃ ┃ ┣ 📜logistic regression_featimp.png
 ┃ ┃ ┣ 📜random forest_featimp.png
 ┃ ┃ ┗ 📜roc_plot.png
 ┣ 📂logs
 ┃ ┗ 📜churn_library.log
 ┣ 📂models
 ┃ ┣ 📜logistic_model.pkl
 ┃ ┗ 📜rfc_model.pkl
 ┣ 📜README.md
 ┣ 📜churn_library.py
 ┣ 📜churn_notebook.ipynb
 ┣ 📜churn_script_logging_and_tests.py
 ┣ 📜constants.py
 ┗ 📜requirements_py3.8.txt

Below is the description of the folder structure:
1. data: contains the raw data file, on which the model is trained
2. images: contains the eda as well as model diagnostic plots
3. logs: contains the log messages generated while running unit tests
4. models: contains the trained model files
5. `churn_library.py`: is the driver program to read, perform eda, feature engineer and train models on the data.
6. `constants.py`: defines paths, hyperparameter and relevant feature names
7. `churn_script_logging_and_tests.py`: contains the unit tests and logging for testing the `churn_library.py`

## Running Files
To run this project need to first install dependencies. Create a virtual environment with `python3.8` using the `requirements_py3.8.txt`.

Then, run the `churn_library.py` as follows:
```shell
python churn_library.py
```

This will populate the following folders:
1. images: this will contain the eda plots in the `eda` subdirectory and model diagnostic plots in the `results` subdirectory
2. models: this will contain the trained models as pickle files


To test this project you need to run `churn_script_logging_and_tests.py` as follows:

```shell
python churn_script_logging_and_tests.py
```

This will populate the following folders:
1. images: this will contain the eda plots in the `eda` subdirectory and model diagnostic plots in the `results` subdirectory
2. models: this will contain the trained models as pickle files
3. logs: this will contain the log messages for the test cases run in the file



