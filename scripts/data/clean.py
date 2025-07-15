import numpy as np
import pandas as pd
from scripts.config import cnfg


def clean_data(data:pd.DataFrame, nan_col_start:str = "latitude",
               nan_col_end:str = "seaLevel", cols_to_dorp:list[str]=None,
               nan_to_zero_columns:list[str] = None,
               outlier_removal:list[list]=None, 
               )->pd.DataFrame:
    """
    Clean and preprocess the dataset for modeling:
    1) Create a deep copy of the input dataset to preserve the original data.
    2) Fill missing values in the specified columns with 0.
    3) Identify and drop rows with any NaN values between specified columns (inclusive).
    4) Remove predefined columns unrelated to model input.
    5) Drop columns that contain single unique value (not useful for ML algorithm).
    6) Remove outlier rows for specified columns within specified percentile range.
    3) convert all numeric columns to pd.Series compatible float32 for resource 
            saving and scikit learn compatibility (e.g. RegressorChain).

    :param data: Input DataFrame to be cleaned.
    :param nan_col_start: Column name where NaN checking should begin (inclusive).
    :param nan_col_end: Column name where NaN checking should end (inclusive).
    :param cols_to_dorp: Column names to drop from dataframe. Optional. 
                            Defaults to config file values.
    :param nan_to_zero_columns: specify a list of columns where NaN values will be replaced by 0. 
                                    Defaults to config file values.
    :param outlier_removal: list of lists that consist of column name, lower and upper 
                                percentile (float) for outlier row removal.
    :return: A cleaned DataFrame with rows and columns filtered and missing data handled.
    """
    df_new = data.copy(deep=True)
    cols_to_dorp = cols_to_dorp or cnfg["data"]["data_cleaning"]["columns_to_drop"]
    if nan_to_zero_columns is not None:
        nan_to_zero = nan_to_zero_columns
    elif "columns_nan_to_zero" in cnfg["data"]["data_cleaning"]:
        nan_to_zero = cnfg["data"]["data_cleaning"]["columns_nan_to_zero"]
    else:
        nan_to_zero = []
    for column in nan_to_zero:
        df_new[column] = df_new[column].fillna(0)
    nan_rows = df_new[df_new.loc[:, nan_col_start:nan_col_end].isnull().any(axis=1)].index
    df_new = df_new.drop(index=nan_rows)
    df_new = df_new.drop(columns=cols_to_dorp, errors='ignore')

    df_new = df_new.drop(columns=[col for col in df_new.columns 
                                  if df_new[col].nunique(dropna=False) == 1])

    if outlier_removal:
        for  feature, lower, upper in outlier_removal:
            df_new = df_new[(df_new[feature] >= df_new[feature].quantile(lower)) 
                            & (df_new[feature] <= df_new[feature].quantile(upper))]
            
    numeric_cols = df_new.select_dtypes(include=np.number).columns
    df_new[numeric_cols] = df_new[numeric_cols].astype(np.float32)

    return df_new
