import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # pyright: ignore[reportMissingModuleSource]

from scripts.utils.eda import top_correlations
from scripts.config import cnfg

def evaluation_tabel(
        predictions: np.ndarray,
        y_true: pd.DataFrame,
        target_index: list[str] = None,
        original_dataset_correlations: pd.Series= None,
        ) -> pd.DataFrame:
    """
    Get model evaluation table with MAE, RMSE and R2 metrics.
    MAPE calculation excludes all 0 values for True values.
    Optionally add highest correlation scores between target and source features.
    :param y_true: target data with true values.
    :param predictions: model predictions.
    :param target_index: names of target labels. Defaults to values from config file.
    :param original_dataset_correlations: correlation matrix of the 
        source dataset that includes target columns with 2 level multi-index.
    :return: DataFrame with target labels as rows an model evaluation 
                metrics as columns.
    """
    targets = target_index or cnfg["data"]["target_columns"]
    df_metrics = pd.DataFrame(index=targets)
    df_metrics["RMSE"] = np.sqrt(mean_squared_error(y_pred=predictions, 
                                                    y_true=y_true, 
                                                    multioutput="raw_values"))
    df_metrics["MAE"] = mean_absolute_error(y_pred=predictions, y_true=y_true,
                                            multioutput="raw_values")
    df_metrics["R2"] = r2_score(y_pred=predictions, y_true=y_true,
                                multioutput="raw_values")
    non_zero_mask = y_true != 0
    df_metrics["masked_MAPE"] = np.mean(np.abs((y_true[non_zero_mask.all(axis=1)
                                                ] - predictions[np.all(non_zero_mask, axis=1)]) / (
                                                    y_true[non_zero_mask.all(axis=1)])), axis=0
                                                    ) * 100
    if original_dataset_correlations is not None:
        for target in targets:
            target_corrs  = original_dataset_correlations[
                (original_dataset_correlations.index.get_level_values(0) == target)
                  | (original_dataset_correlations.index.get_level_values(1) == target)]
            if target_corrs.empty:
                df_metrics.at[target, "max_corr_w_orig_feat"] = np.nan
            else:
                df_metrics.at[target, "max_corr_w_orig_feat"] = target_corrs.max()                
            
    return df_metrics
