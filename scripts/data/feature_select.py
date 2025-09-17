import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn.base import BaseEstimator, TransformerMixin # pyright: ignore[reportMissingModuleSource]

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
        self.feature_names_in_ = None


    def fit(self, X, y=None):
        """
        Store the original feature names. Required for scikit-learn compatibility.
        Does not perform any fitting operation.
        :param X: Input DataFrame.
        :param y: Ignored.
        :return: self
        """
        self.feature_names_in_ = X.columns.to_list()
        return self


    def transform(self, X: pd.DataFrame, feature_to_drop:list[str]=None)->pd.DataFrame:
        """
        Remove specified features (columns) from the input DataFrame.
        Uses instance-level 'features_to_drop' unless an override list is provided.
        If self.feature_names_in_ are not populated with 'fit' method, stores colimn names.
        :param X: input data.
        :param features_to_drop: List of column names to drop.
        :return: A new pandas DataFrame with the specified features removed.
        """
        if self.feature_names_in_ is None:
            self.feature_names_in_ = X.columns.to_list()

        df_new = X.copy(deep=True)
        features = feature_to_drop or self.features_to_drop
        
        if features:
            df_new = df_new.drop(columns=features, errors='ignore')
            
        return df_new
    