import pandas as pd # pyright: ignore[reportMissingModuleSource]


def top_correlations(data:pd.DataFrame, corr_threshold:float=0.95)->pd.Series:
    """
    Identify and return highly correlated feature pairs above a specified threshold.
    :param data: pandas DataFrame containing the input dataset. Defaults to df_clean.
    :param corr_threshold: Correlation threshold to filter feature pairs. Default is 0.95.
    :return: A pandas Series with multi-index (feature pairs) and correlation values.
    """
    corrs = data.select_dtypes(include='number').corr()
    corrs = corrs.unstack().sort_values(ascending=False)
    corrs = corrs[(corrs.abs() > corr_threshold) & 
                        (corrs.index.get_level_values(0) != corrs.index.get_level_values(1))]
    corrs = corrs.drop_duplicates()
    return corrs
