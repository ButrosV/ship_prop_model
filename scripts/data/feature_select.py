import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from scripts.config import cnfg


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Remove selected features (columns) from the input DataFrame.
    Inherits from scikit-learn's BaseEstimator and TransformerMixin 
            for scikit-learn pipeline compatibility.
     Used to simplify datasets by excluding specified columns during preprocessing.
    :param data: pandas DataFrame to process.
    :param features_to_drop: List of column names to drop.
        Defaults to config file values.
    """

    def __init__(self, features_to_drop:list=None):
        """"
        Initialize with the provided data and feature configurations.
        Store the list of features to drop for later use in transformation.
        """
        self.features_to_drop = features_to_drop or cnfg[
            "data"]["data_preprocessing"]["collinear_features"]


    def fit(self, X, y=None):
        """
        Fit method for compatibility with scikit-learn pipelines.
        Does not perform any fitting operation.
        """
        return self
    

    def transform(self, X: pd.DataFrame, feature_to_drop:list=None)->pd.DataFrame:
        """
        Remove specified features (columns) from the input DataFrame.
        Uses instance-level 'features_to_drop' unless an override list is provided.
        :param X: input data.
        :param features_to_drop: List of column names to drop.
        :return: A new pandas DataFrame with the specified features removed.
        """
        df_new = X.copy(deep=True)
        features = feature_to_drop or self.features_to_drop
        if features:
            df_new = df_new.drop(columns=features)
            
        return df_new
    