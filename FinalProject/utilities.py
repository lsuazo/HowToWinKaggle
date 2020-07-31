import os
import numpy as np
import pandas as pd

def load_and_preprocess(parse_date = True):
    data_fol = 'data'
    sales = pd.read_csv(os.path.join(data_fol, 'sales_train.csv'))
    if parse_date:
        date_col = pd.to_datetime(sales.date, format='%d.%m.%Y')
        sales.date = date_col
        sales['Month'] = sales.date.dt.to_period('M')
    return sales


def aggregate_monthly(sales_df):
    assert 'Month' in sales_df.columns, 'Month column not found'
    return sales_df.groupby(['Month', 'shop_id', 'item_id']).agg({'item_cnt_day': np.sum, 'item_price': np.mean}).rename(columns={'item_cnt_day':'item_cnt_month'}).reset_index()


def merge_pred_with_test(prediction, fill_val=0, pred_col='item_cnt_month'):
    assert pred_col in prediction.columns, f'Couldnt find {pred_col}'
    data_fol = 'data'
    test = pd.read_csv(os.path.join(data_fol, 'test.csv'))
    merged =  pd.merge(test, prediction, on=['shop_id', 'item_id'], how='left').fillna(fill_val)
    if pred_col != 'item_cnt_month':
        merged = merged.rename(columns={pred_col:'item_cnt_month'})
    return merged.set_index('ID')['item_cnt_month']