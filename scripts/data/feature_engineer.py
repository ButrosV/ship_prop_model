import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from scripts.config import cnfg

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Engineer features: angular, force interaction and time.
    Inherits from scikit-learn's BaseEstimator and TransformerMixin 
            for scikit-learn pipeline compatibility.
    :param angular_features: List of angle-based feature column names (in degrees).
                                Defaults to config file values.
    :param env_force_features: List of (angle feature, force feature) pairs for decomposing forces.
                                Defaults to config file values.
    :param base_angle_feats: Reference angle column name (default, 'heading' from config file).
    :param base_time_feats: Column name for timestamp (default, 'timestamp' from config file).
    :param drop_original_features: Flag to optionally remove original features a self.base_angle_feat 
                                    class level with 'transform()' method.
    """


    def __init__(self,
                 angular_features: list[str] = None,
                 env_force_features: list[list] = None,
                 base_angle_feats: str = None,
                 base_time_feats: str = None,
                 drop_original_features: bool = False
                 ):
        """
        Initialize with the provided data and feature configurations.
        Store configuration parameters for use in transformations.
        """
        self.angular_features = angular_features or cnfg["feature_engineering"]["angular_features"]
        self.env_force_features = env_force_features or cnfg["feature_engineering"][
            "environment_forces_features"]
        self.base_angle_feats = base_angle_feats or cnfg["feature_engineering"]["base_angle_feature"]
        self.base_time_feats = base_time_feats or cnfg["feature_engineering"]["base_time_feature"]
        self.drop_original_features = drop_original_features
        self._transformed_feature_names = None


    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit method for compatibility with scikit-learn pipelines.
        Does not perform any fitting operation.
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = None

        return self
    
    
    @property
    def feature_names_out_(self):
        """
        Getter for feature names after transformations.
        If transformations have been applied, return the column names.
        """
        return self._transformed_feature_names
    

    @staticmethod
    def compute_relative_angle(angle1, angle2):
        """
        Compute absolute minimal angle difference between two angles (in degrees).
        :param angle1: First angle or Series of angles.
        :param angle2: Second angle or Series of angles.
        :return: Absolute minimal angular difference in degrees.
        """
        return np.abs(((angle1 - angle2 + 180) % 360) - 180)
    

    @staticmethod
    def drop_columns(data:pd.DataFrame, drop_list:list[str]):
        """
        Drop specified columns from the DataFrame.
        Check for presence of 'drop_list' features in DataFrame.
        :param data: DataFrame to modify.
        :param drop_list: List of column names to drop.
        :return: Modified DataFrame with columns removed.
        """
        data.drop(
            columns=[col for col in drop_list if col in data.columns],
            errors='ignore',
            inplace=True
            )
        
        return data


    def transform_angles(self,
                        X: pd.DataFrame,
                        angular_feature: list[str] = None,
                        drop_source_feature: bool = False) -> pd.DataFrame:
        """
        Compute relative angles between angular features and base_angle_feat.
        This feature provides if various external forces (current, wind, etc) facilitate 
            or counteract ship's course.
        If not specified otherwise, use parameters defined at 'FeatureEngineer' initialization.
        Check for duplicated columns and drop if any are created via multiple method calls.
        Optionally drop original columns.
        :param X: input data.
        :param angular_feature: List of angle-based feature column names (in degrees).
        :param drop_source_feature: Flag to drop original angle columns (default, False).
        :return: DataFrame with added relative angle features.
        """
        df_angle = X.copy(deep=True)
        features = angular_feature or self.angular_features

        for angle_feature in features:
            if angle_feature in df_angle.columns\
                and self.base_angle_feats in df_angle.columns:
                angle_to_heading = self.compute_relative_angle(
                    df_angle[angle_feature], df_angle[self.base_angle_feats])

                df_angle.insert(loc=df_angle.columns.get_loc(angle_feature) + 1,
                                column=angle_feature + '_rel_head',
                                value=angle_to_heading,
                                allow_duplicates=True)
                
        df_angle = df_angle.loc[:, ~df_angle.columns.duplicated()]  
        if drop_source_feature:
            df_angle = self.drop_columns(data=df_angle, drop_list=features+[self.base_angle_feats])
   
        return df_angle


    def transform_force_components(self,
                                   X: pd.DataFrame,
                                   env_force_feature: list[tuple] = None,
                                   drop_source_feature: bool = False) -> pd.DataFrame:
        """
        Add force features based on relative angle to base_angle_feat.
        This feature provides index for external forces (current, wind, etc) speed and angle
             counteracting ship's course. Higher index means stronger tail/cross components.
        If not specified otherwise, use parameters defined at 'FeatureEngineer' initialization.
        Check for duplicated columns and drop if any are created via multiple method calls.
        Optionally drop original columns.
        :param X: input data.
        :param env_force_feature: List of (angle_col, force_col) pairs for decomposing forces.
        :param drop_source_feature: Flag to drop original force columns (default, False).
        :return: DataFrame with new force features.
        """
        df_force = X.copy(deep=True)
        features = env_force_feature or self.env_force_features

        for angle_col, force_col in features:
            if angle_col in df_force.columns\
            and force_col in df_force.columns\
            and self.base_angle_feats in df_force.columns:
                rel_angle_rad = np.radians(
                    self.compute_relative_angle(
                        df_force[angle_col], df_force[self.base_angle_feats]))
                idx_force = df_force.columns.get_loc(force_col)

                df_force.insert(loc=idx_force + 1,
                                column=force_col + "_head",
                                value=df_force[force_col] * np.cos(rel_angle_rad),
                                allow_duplicates=True)

                # Insert cross-wind/current component
                df_force.insert(loc=idx_force + 1,
                                column=force_col + "_cross",
                                value=df_force[force_col] * np.sin(rel_angle_rad),
                                allow_duplicates=True)
            
                if drop_source_feature:
                    df_force = self.drop_columns(data=df_force, drop_list=[angle_col, force_col])
            
        df_force = df_force.loc[:, ~df_force.columns.duplicated()]
                
        return df_force
    

    def eng_direction_features(self,
                               X: pd.DataFrame, 
                                angular_feature: list[str] = None,
                                drop_source_feature: bool=False
                                )->pd.DataFrame:
        """
        Engineer sine and cosine components of directional features to preserve circularity.
        New sine and cosine features address issue arising from circularity of angle features
                     - e.g. that after 360 comes 0.
        If not specified otherwise, use parameters defined at 'FeatureEngineer' initialization.
        Check for duplicated columns and drop if any are created via multiple method calls.
        Optionally drop original columns.
        :param X: input data.
        :param angular_feature: List of directional feature columns (in degrees).
        :param drop_source_feature: Flag to drop original direction columns (default, False).
        :return: DataFrame with sine and cosine of directional features.
        """
        df_direction = X.copy(deep=True)
        features = angular_feature or self.angular_features

        for angle_feature in features:
            if angle_feature in df_direction.columns:
                orig_feat_index = df_direction.columns.get_loc(angle_feature)
                df_direction.insert(loc=orig_feat_index+1,
                                    column=angle_feature+'_sin',
                                    value=np.sin(2 * np.pi * df_direction[angle_feature] / 360),
                                    # value=np.sin(np.radians(df_direction[feature])),
                                    allow_duplicates=True)
                df_direction.insert(loc=orig_feat_index+1,
                                    column=angle_feature+'_cos',
                                    value=np.cos(2 * np.pi * df_direction[angle_feature] / 360),
                                    # value=np.cos(np.radians(df_direction[feature])),
                                    allow_duplicates=True)

        if drop_source_feature:
            df_direction = self.drop_columns(data=df_direction, drop_list=features)
        df_direction = df_direction.loc[:, ~df_direction.columns.duplicated()]

        return df_direction
    

    def eng_circular_time_feats(self,
                                X: pd.DataFrame,
                                base_time_feat: str = None,
                                ):
        """
        Create hour and month features from timestamp. 
        Both potentially indicate daily or seasonal weather patterns 
                    affecting ship's speed and engine workloads.
        Calculate sines and cosines to preserve circularity - e.g. that after 23 or 12 comes 0/1.
        If not specified otherwise, use parameters defined at 'FeatureEngineer' initialization.
        Convert timestamp column to data type if required.
        Check for duplicated columns and drop if any are created via multiple method calls.
        Drop original column as most models will not process timestamp objects.
        :param X: input data.
        :param base_time_feat: Column name containing time information (default, None).
        :param drop_source_feature: Flag to drop original timestamp columns (default, False).
        :return: DataFrame with sine and cosine components of the hour feature.
        """
        df_time = X.copy(deep=True)
        time_feature = base_time_feat or self.base_time_feats
        if time_feature in df_time.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_time[time_feature]):
                df_time[time_feature] = pd.to_datetime(df_time[time_feature], errors='coerce')
            idx_time = df_time.columns.get_loc(time_feature)
            new_time_features = [("hours", df_time[time_feature].dt.hour, 24),
                                 ("months", df_time[time_feature].dt.month, 12)]

            for name, values, circular_divisor in new_time_features:
                df_time.insert(loc=idx_time+1,
                                    column=name + "_sin",
                                    value=np.sin(2 * np.pi * values / circular_divisor),
                                    allow_duplicates=True)
                df_time.insert(loc=idx_time+1,
                                    column=name + "_cos",
                                    value=np.cos(2 * np.pi * values / circular_divisor),
                                    allow_duplicates=True)
            df_time = self.drop_columns(df_time, [time_feature])

        return df_time


    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to angles, forces, directions, and power-to-speed ratio.
        Calls all 'FeatureEngineer' feature engineering steps sequentially and removes duplicates.
        Uses 'FeatureEngineer' instance-level 'drop_original_features' flag to remove original features.
        :param X: input data.
        :return: Transformed DataFrame with engineered features.
        """
        if not hasattr(X, "columns"):
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        elif isinstance(X, pd.DataFrame):
            df = X.copy(deep=True)
        df = self.transform_angles(X=df)
        df = self.transform_force_components(X=df, drop_source_feature=self.drop_original_features)
        df = self.eng_direction_features(X=df, drop_source_feature=self.drop_original_features)
        df = self.eng_circular_time_feats(X=df)
        if self.drop_original_features is not False:
            df = self.drop_columns(data=df, drop_list=[self.base_angle_feats])
        df = df.loc[:, ~df.columns.duplicated()]
        self._transformed_feature_names = df.columns.tolist()

        return df
    