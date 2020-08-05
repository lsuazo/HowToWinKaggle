import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

###### Reading and simple processing

def load_and_preprocess(parse_date = True):
    data_fol = 'data'
    sales = pd.read_csv(os.path.join(data_fol, 'sales_train.csv'))
    if parse_date:
        date_col = pd.to_datetime(sales.date, format='%d.%m.%Y')
        sales.date = date_col
        sales['Month'] = sales.date.dt.to_period('M')
    return sales


def merge_pred_with_test(prediction, fill_val=0, pred_col='item_cnt_month'):
    assert pred_col in prediction.columns, f'Couldnt find {pred_col}'
    data_fol = 'data'
    test = pd.read_csv(os.path.join(data_fol, 'test.csv'))
    merged =  pd.merge(test, prediction, on=['shop_id', 'item_id'], how='left').fillna(fill_val)
    if pred_col != 'item_cnt_month':
        merged = merged.rename(columns={pred_col:'item_cnt_month'})
    return merged.set_index('ID')['item_cnt_month']


####### Data preparation

PRED_MONTH_COL_NAME = 'pred_month'

def aggregate_monthly(sales_df):
    assert 'Month' in sales_df.columns, 'Month column not found'
    return sales_df.groupby(['Month', 'shop_id', 'item_id']).agg({'item_cnt_day': np.sum, 'item_price': np.mean}).rename(columns={'item_cnt_day':'item_cnt_month'}).reset_index()


def create_windowed_XY(df, fit_cols, append_pred_month=False, append_item_id=False, append_shop_id=False):
    total_pred_months = df.shape[1] - fit_cols
    if total_pred_months <= 0:
        return (None, None)
    
    Xs = []
    Ys = []
    for i in range(0,total_pred_months):
        X = df.iloc[:,i:fit_cols+i].copy()
        X.columns = range(fit_cols,0,-1)
        Y = df.iloc[:,fit_cols+i].copy()
        if append_pred_month:
            X[PRED_MONTH_COL_NAME] = Y.name.month
        Y = Y.rename('Y')
        
        Xs.append(X)
        Ys.append(Y)

    Y = pd.concat(Ys).reset_index(drop=True)
    X = pd.concat(Xs).reset_index(drop=True)
    return X,Y


def cross_val_score(df, model, fit_months=24, strides=3, add_pred_month=False, X_func_list=None, train_final=False):
    
    total_months = df.shape[1]
    fold_size = fit_months + strides + 1

    in_sample_errors = []
    validate_errors = []
    oos_months = []
    
    for i in range(total_months - fold_size + 1):
        sub_df = df.iloc[:, i:i+fold_size]
        oos_month = sub_df.columns[-1]
        oos_months.append(oos_month)
        print(f'working on iteration: {i}, oos month: {oos_month}')
        
        X_train, Y_train = create_windowed_XY(sub_df.iloc[:,:-1], fit_months, add_pred_month)
        X_validate, Y_validate = sub_df.iloc[:,-1-fit_months:-1].copy(), sub_df.iloc[:,-1].copy()
        X_validate.columns = range(fit_months,0,-1)
        if add_pred_month:
            X_validate[PRED_MONTH_COL_NAME] = Y_validate.name.month
        Y_validate = Y_validate.rename('Y')
        
        if X_func_list is not None:
            for func in X_func_list:
                X_train, X_validate = func(X_train), func(X_validate)

        fit_model = model.fit(X_train, Y_train)

        Y_train_pred = fit_model.predict(X_train)
        in_sample_error = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
        in_sample_errors.append(in_sample_error)
        
        Y_validate_pred = fit_model.predict(X_validate)
        valid_error = np.sqrt(mean_squared_error(Y_validate, Y_validate_pred))
        validate_errors.append(valid_error)
        
    final_model = None
    if train_final:
        X_train, Y_train = create_windowed_XY(df.iloc[:,-(fold_size-1):], fit_months)
        final_model = model.fit(X_train, Y_train)
    
    errors = pd.DataFrame({'oos_month':oos_months, 'in_sample': in_sample_errors, 'out_of_sample':validate_errors}).set_index('oos_month')
    return errors, final_model
   
    