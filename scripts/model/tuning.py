from pathlib import Path
import joblib
import numpy as np
from scipy.stats import uniform, randint, norm

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor

from scripts.data.feature_select import FeatureSelector
from scripts.data.feature_engineer import FeatureEngineer

from scripts.config import cnfg

class HyperParamSearch():
    """
    A class for conducting hyperparameter optimization on multi-output models
    using RegressorChain. Includes preprocessing, randomized search, grid search,
    and automatic parameter sampling from normal-like distributions.

    Attributes:
        model (RegressorChain): The model wrapped in a RegressorChain.
        folder_path (str): Path to the folder where models and preprocessors are saved.
        preprocess_output (Pipeline or None): Stores the preprocessing pipeline after fitting.
        random_search_result (RandomizedSearchCV or None): Stores the result of the random search.
        grid_search_result (GridSearchCV or None): Stores the result of the grid search.
        new_grid_params (dict or None): Stores newly generated grid parameters from normal distribution.
    """


    def __init__(self, model:Path=None, save_folder:str=None, remove_features:list[str]=None):
        """
        Initialize the hyperparameter search with optional custom model and save directory.
        Create save directory if it does not exist.
        :param model: Custom multi-output model. Defaults to RegressorChain(XGBRegressor()).
        :param save_folder: Path to save/load model and preprocessor files. Defaults to model
                directory from config file.
        :param remove_features: optional list of features to be removed during preprocessing. 
                            If none, defaults to collinear feature list from config file.
        """
        self.model = model or RegressorChain(XGBRegressor(n_jobs=1))
        self.folder_path = save_folder or Path(__file__).parent.parent / cnfg["models"]["model_dir"] # for script
        # self.folder_path = save_folder or Path.cwd() / cnfg["models"]["model_dir"]"  # for jupyter
        self.preprocess_output = None
        self.random_search_result = None
        self.grid_search_result = None
        self.new_grid_params = None
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.remove_features = remove_features
        

    def preprocess(self, X, preprocess_filename:str=None, preproces_pipeline=None,):
        """
        Preprocess the input data using feature selection, feature engineering, and scaling.
        If preprocessor pipeline not provided, constructs preprocessing Pipeline with 
            custom FeatureSelector and FeatureEngineer classes with default config file 
            defined feature processing features except for dropping source features after processing.
        Save or load the pipeline from disk to avoid refitting.
        :param X: The input features as a pandas DataFrame or NumPy array.
        :param preprocess_filename: Name of the file for storing/loading the pipeline.
        :param preprocess_pipeline: Optional custom preprocessing pipeline to use.
        :return: Fitted preprocessing pipeline object.
        """
        filename = preprocess_filename or cnfg["data"]["data_preprocessing"]["preprocessor_file"]
        preprocessor_path = self.folder_path / filename
        if preprocessor_path.is_file():
            preprocess_all = joblib.load(preprocessor_path)
        else:
            preprocess_all = preproces_pipeline or Pipeline([
                    ("select_features", FeatureSelector(features_to_drop=self.remove_features)),
                    ("engineer_features", FeatureEngineer(drop_original_features=True,)),
                    ("scaler", StandardScaler()),
                    ])
            preprocess_all.fit(X)
            joblib.dump(value=preprocess_all, filename=preprocessor_path)
        self.preprocess_output = preprocess_all
        return preprocess_all


    def normal_distr_parameters(self, params_centers, 
                                range_from_mean: float = 0.1, 
                                n_params: int = 5)->dict:
        """
        Generate parameter values for each key in a dictionary using an approximation
        of a normal distribution centered around the median value.
        Generate parameter values around each mean with a normal distribution.
        Get data points within distance of 3 std from the new data point mean.
        Integer values are rounded and kept as int type.
        :param params_centers: A dictionary with numeric values. Dictionary keys must follow 
                                format 'base_estimator__[valid_model_param]'
                                (e.g. RandomizedSearchCV '.best_params_' output.)
        :param range_from_mean: Fraction of the new mean to define range from which
                                standard deviation is estimated (±range = mean * fraction).
        :param n_params: Number of percentile-based values to sample from the normal 
                        distribution.
        :return: A dictionary where each key maps to a list of unique sampled values
                (as int or float), representing approximate normal distribution samples.
        """
        percentiles = np.linspace(0,1, n_params)[1:-1]
        if isinstance(params_centers, dict):
            new_params = params_centers.copy()
        else:
            raise ValueError("Parameter centers (params_centers) must be a dictionary.")
        
        for key, mean in new_params.items():
            range_factoor = mean * range_from_mean
            std_approx = (range_factoor * 2) / 6
            
            data_points = np.append(
                norm.ppf(percentiles, loc=mean, scale=std_approx), 
                [mean - range_factoor, mean, mean + range_factoor]
                )
            if isinstance(new_params[key], int):
                new_params[key] = list(set(
                    int(np.floor(point)) if point <= mean \
                    else int(np.ceil(point)) for point in data_points))
            else:
                new_params[key] = list(set(data_points))
                
        self.new_grid_params = new_params
        return new_params
    
    
    def random_search(self, X, y, rs_model_filename:str=None, param_distribution=None,
                      n_iter:int=200, cv:int=3, n_jobs:int=4, scoring:str='r2'):
        """
        Perform a randomized search over hyperparameters using cross-validation.
        If a saved model is available, it is loaded instead of retraining.
        If necessary, run data preprocessing.
        :param X: Training feature data.
        :param y: Training target data.
        :param rs_model_filename: Filename for storing/loading the RandomizedSearchCV model.
        :param param_distribution: Dictionary of parameter distributions for sampling.
                If not provided, generates param_distributions from ranges defined in config file.
        :param n_iter: Number of parameter settings sampled.
        :param cv: Number of cross-validation folds.
        :param n_jobs: Number of parallel jobs to run.
        :param scoring: Scoring metric for model evaluation.
        :return: Fitted RandomizedSearchCV object.
        """
        filen = rs_model_filename or cnfg["hyperparameter_tuning"]["random_search_file"]
        rs_model_path = self.folder_path / filen
        if self.preprocess_output is None:
            self.preprocess(X)  
        if rs_model_path.is_file():
            random_search = joblib.load(filename=rs_model_path)
            print("Loading Random search results from saved model.")  # remove after testing
        else:
            print("Fitting and saving new random search.")  # remove after testing
            ranges = cnfg["hyperparameter_tuning"]["random_search_ranges"]
            param_distributions = param_distribution or {
                key:(randint(value[0], value[1]
                             ) if isinstance(value[0], int) and isinstance(value[1], int)
                               else uniform(value[0], value[1])) 
                               for key, value in ranges.items()}
            random_model = self.model
            random_search = RandomizedSearchCV(
                estimator = random_model,
                param_distributions = param_distributions,
                n_iter=n_iter,
                n_jobs=n_jobs,
                scoring=scoring,
                verbose=2,  #remove after testing
            )          
            X_train_prep = self.preprocess_output.transform(X)
            random_search.fit(X_train_prep, y)
            if hasattr(random_search, "best_estimator_"):  # remove after testing
                joblib.dump(value=random_search, filename=rs_model_path)
            else:
                print("No best estimator found, random search likely failed.")
        self.random_search_result = random_search
        return random_search
        

    def grid_search(self, X, y, gs_model_filename:str=None, param_grid=None,
                      cv:int=3, n_jobs:int=4, scoring:str='r2'):
        """
        Perform grid search using parameters from a distribution or previous random search.
        If a saved model is available, it is loaded instead of retraining.
        If necessary, run data preprocessing.
        :param X: Training feature data.
        :param y: Training target data.
        :param gs_model_filename: Filename to save/load the GridSearchCV model.
        :param param_grid: Parameter grid to use. If None, will generate using best random search params.
        :param cv: Number of cross-validation folds.
        :param n_jobs: Number of parallel jobs.
        :param scoring: Scoring metric for model evaluation.
        :return: Fitted GridSearchCV object.
        """
        filen = gs_model_filename or cnfg["hyperparameter_tuning"]["grid_search_file"]
        gs_model_path = self.folder_path / filen
        if self.preprocess_output is None:
            self.preprocess(X) 
        if gs_model_path.is_file():
            grid_search = joblib.load(filename=gs_model_path)
            print("Loading gridsearch results from saved model.")
        else:
            print("Fitting and saving new gridsearch.")
            if self.random_search_result and param_grid is None:
                self.random_search(X, y)
            new_grid = param_grid or self.normal_distr_parameters(
                params_centers=self.random_search_result.best_params_)
            grid_model = self.model
            grid_search = GridSearchCV(estimator=grid_model,
                            param_grid=new_grid,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=2,  # remove after testting
                            )
            X_train_prep = self.preprocess_output.transform(X)
            grid_search.fit(X_train_prep, y)
            if hasattr(grid_search, "best_estimator_"):  # remove after testing
                joblib.dump(value=grid_search, filename=gs_model_path)
            else:
                print("No best estimator found, gridsearch likely failed.")
        self.grid_search_result = grid_search
        return grid_search
        

    def full_param_search(self, X, y, param_distribution=None, param_grid=None, 
                    grid_param_range:float=0.1, grid_n_params:int=5,
                    n_iter:int=200, random_cv:int=3, grid_cv:int=3, n_jobs:int=4,
                    scoring:str='r2'):
        """
        Conduct full hyperparameter search: randomized search followed by grid search.
        Grid search parameters are generated as a normal-like distribution from best 
            results of the random search.
        :param X: Training feature data.
        :param y: Training target data.
        :param param_distribution: Parameter distribution for the random search.
        :param param_grid: Parameter grid for grid search. If None, it will be generated.
        :param grid_param_range: Range from mean to define std dev for grid param generation.
        :param grid_n_params: Number of values per parameter to sample in grid.
        :param n_iter: Number of iterations for randomized search.
        :param random_cv: Number of cross-validation folds in randomized search.
        :param grid_cv: Number of cross-validation folds in grid search.
        :param n_jobs: Number of parallel jobs to run.
        :param scoring: Scoring metric for evaluation.
        :return: Final GridSearchCV object fitted on data.
        """
        self.random_search(X=X, y=y, param_distribution=param_distribution, n_iter=n_iter,
                           cv=random_cv, n_jobs=n_jobs, scoring=scoring)
        normal_grid = param_grid or self.normal_distr_parameters(
                params_centers=self.random_search_result.best_params_,
                range_from_mean=grid_param_range, n_params=grid_n_params)
        self.grid_search(X=X, y=y, param_grid=normal_grid, cv=grid_cv, n_jobs=n_jobs,
                         scoring=scoring)
        return self.grid_search_result.best_estimator_
    