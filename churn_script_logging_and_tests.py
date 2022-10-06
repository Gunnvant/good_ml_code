import os
import logging
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_data_exists():
    '''
    test existence of data file.
    '''
    try:
        assert os.path.exists(cls.constants.PATH_DATA)
        logging.info('File exists:SUCCESS')
    except FileExistsError:
        logging.error('File exists:FAILED')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    df = cls.import_data(cls.constants.PATH_DATA)
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info('File read : SUCCESS')
    except AssertionError as err:
        logging.error("File read : FAILED")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = cls.import_data(cls.constants.PATH_DATA)
    cls.create_dependent_var(df)
    try:
        cls.perform_eda(df)
        logging.info('EDA function ran: SUCCESS')
    except AssertionError:
        logging.error('EDA function: FAILED')

    eda_files = ['churn_hist.png', 'cust_age_hist.png',
                 'heatmap.png', 'marital_status.png',
                 'total_trans.png']
    for file in eda_files:
        path_file = os.path.join(cls.constants.PATH_IMG_EDA, file)
        try:
            assert os.path.exists(path_file)
            logging.info(f'File {file} exists : SUCCESS')
        except AssertionError:
            logging.error(f'File {file} exists : FAILED')


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = cls.import_data(cls.constants.PATH_DATA)
    cls.create_dependent_var(df)
    cat_list = cls.constants.CATEGORY_LIST
    cls.encoder_helper(df, cat_list)
    for cat_col in cat_list:
        col_name = cat_col + "_Churn"
        try:
            assert col_name in df.columns
            logging.info(f"Column {cat_col} encoded : SUCCESS")
        except AssertionError:
            logging.error(f"Column {cat_col} encoded : FAILED")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df = cls.import_data(cls.constants.PATH_DATA)
    cls.create_dependent_var(df)
    cat_list = cls.constants.CATEGORY_LIST
    cls.encoder_helper(df, cat_list)
    try:
        X_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)
        logging.info('Feature engineering : SUCCESS')
    except AssertionError:
        logging.info('Feature engineering : FAILED')
    try:
        assert X_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info('X,y train/test have data : SUCCESS')
    except AssertionError:
        logging.info('X,y train/test are empty : FAILED')


def test_train_models():
    '''
    test train_models
    '''
    df = cls.import_data(cls.constants.PATH_DATA)
    cls.create_dependent_var(df)
    cat_list = cls.constants.CATEGORY_LIST
    cls.encoder_helper(df, cat_list)
    X_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)
    try:
        cls.train_models(X_train, x_test, y_train, y_test, df)
        logging.info('Models trained : SUCCESS')
    except AssertionError:
        logging.error('Models trained : FAILED')
    res_plots = [
        'feature_imp.png',
        'logistic regression_featimp.png',
        'random forest_featimp.png',
        'roc_plot.png']
    for plot in res_plots:
        path_plot = os.path.join(cls.constants.PATH_IMG_RES, plot)
        try:
            assert os.path.exists(path_plot)
            logging.info(f'Plot {plot} exists : SUCCESS')
        except AssertionError:
            logging.error(f'Plot {plot} exists : FAILED')
    model_files = ['logistic_model.pkl', 'rfc_model.pkl']
    for model in model_files:
        path_model = os.path.join(cls.constants.PATH_MODELS, model)
        try:
            assert os.path.exists(path_model)
            logging.info(f'Model {model} exists : SUCCESS')
        except AssertionError:
            logging.error(f'Model {model} exists : FAILED')


if __name__ == '__main__':
    test_data_exists()
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
