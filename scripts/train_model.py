import pandas as pd
import numpy as np
import random
from scipy.stats import uniform, randint, norm

from pathlib import Path
import gdown
import joblib

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split,  GridSearchCV, \
    RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.multioutput import RegressorChain

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # SAMPLE main.py
# from scripts.data_loader import load_data
# from scripts.preprocessor import preprocess_data
# from scripts.model import train_model, evaluate_model


# def main():
#     # Load data
#     X_train, X_test, y_train, y_test = load_data("data/input.csv")

#     # Preprocess
#     X_train_pre, X_test_pre = preprocess_data(X_train, X_test)

#     # Train
#     model = train_model(X_train_pre, y_train)

#     # Evaluate
#     metrics = evaluate_model(model, X_test_pre, y_test)
#     print(metrics)

# if __name__ == "__main__":
#     main()


print("ok")