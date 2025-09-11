# Ship Propulsion Prediction
***
## Task
Implement machine learning pipeline and deploy it via API to predict vessel **shaftPower** and **speedOverGround** from environmental and engine data, including full support for data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.
The challenge is to choose the most appropriate features, model and parameters as well as to deploy trained model into a FastAPI application.

## Description
To reach project goals:
#### 1. Develop a machine learning model that predicts two target variables:
* `shaftPower`
* `speedOverGround`

using input features from Ship Power Prediction Dataset:
* A):
    * cleaning, engineering and preprocessing from all 39 features of the source dataset.
* B):
    * selecting and engineering narrower, task specific, features:
        * **Weather data**: `windSpeed`, `windDirection`, `waveHeight`, `waveDirection`, `swellHeight`, `swellDirection`, `currentSpeed`, `currentDirection`, `airTemperature`
        * **Engine data**: `mainEngineMassFlowRate`

##### 1.1. Data Preprocessing

* Load `.json` dataset into a pandas `DataFrame`
* Handle missing values (e.g., replace NaNs with 0s or drop rows)
* Normalize features using `StandardScaler`
* Drop irrelevant or highly collinear features based on configuration
* Optional: use `--task_features` flag to train only on features mentioned in the task
* Use the `FeatureSelector` class to dynamically remove features based on task flags or correlation thresholds.

##### 1.2. Feature Engineering

Implemented via a custom `FeatureEngineer` class:
* For case A) - all features utilized:
    * Angular decomposition of direction-based features (e.g., wind/heading)
    * Force vector breakdowns for environment-driven features
    * Time-based and derived metrics
    * adjust time and directional features for preserving circularity (cos/sin)
* For case B) - narrower, task specific, features set used:
    * adjust directional features for preserving circularity (cos/sin)

##### 1.3. Model Selection
See model comparison in `.\notebooks\ship_hw_EDA_experiments.ipynb`.

Final setup used:
* `RegressorChain` wrapper with:
  * XGBoost (`XGBRegressor`)
* Option to tune using:
  * `RandomizedSearchCV` + `GridSearchCV`
  * Auto-sampled hyperparameter ranges for`GridSearchCV` from normal distributions around `RandomizedSearchCV` best results.

##### 1.4. Evaluation
Metrics used to compare models for model selection:
* R² Score
* RMSE
* MAE
* MAPE (masked for 0s)
* Highest raw data Feature–target correlation score for baseline

### 2: FastAPI Deployment

Test api with the same input feature count as you trained your model with.

## Project Structure

```
.
├───main.py  # Entry point (train, evaluate, tune)
├───api
|   ├───app.py  # FastAPI application
│   └───schema.py # Pydantic model for input validation
├───data  # Raw dataset files
├───models  # Saved models and preprocessors
├───notebooks
│   └───ship_hw_EDA_experiments.ipynb
├───scripts
│   ├───config.py
│   ├───data  # Data cleaning, loading, feature prep
│   │   ├───clean.py
│   │   ├───split.py
│   │   ├───feature_engineer.py
│   │   ├───feature_select.py
│   │   └───load.py
│   ├───model # Model training, tuning, evaluation
│   │   ├───evaluation.py
│   │   ├───get_model.py
│   │   └──tuning.py
│   └───utils  # EDA tools (e.g. correlation heatmap)
│       ├───eda.py
│       └───visual.py
├───config.yaml
├───README.md
├───.gitignore
└───requirements.txt
```

## Usage

### Install Dependencies
Clone the [git repository](https://github.com/ButrosV/ship_prop_model.git) or download `.\notebooks\ship_hw_EDA_experiments.ipynb`.

```bash
pip install -r requirements.txt
```

### Run Model Pipeline

Train and evaluate the model using default model and config file hyperparameters:

```bash
python train_model.py
```

Train with full hyperparameter tuning (\~15 hrs):

```bash
python train_model.py --tune_model
```

Train using only task-specified features:

```bash
python train_model.py --task_features
```

Combine both:

```bash
python train_model.py --tune_model --task_features
```

### Run Application
```bash
uvicorn api.app:app --reload
```
## Configuration

YAML config file includes:
* Paths to model and data directories 
* Default preprocessor and model file names
* Default model parameters for final `XBRegressor()` or for tuning initialization with `RandomizedSearchCV`.
* Feature drop lists
* Deafault column name definitions for targets and feature engineering defaults

## Justification for Design Choices

* **RegressorChain** allows leveraging multi-output dependencies (shaftPower ↔ speedOverGround).
* **XGBoost** provides:
    * Acceptable evaluation metric performance.
    * Less costly in terms of computational resources (training times and RAM requirements) when compared to `RandomForest`.
    * Thus more suitable for hyperparameter tuning.
* Custom feature engineering enables richer directional awareness, interactions among forces and potential seasonal weather patterns.
* Modular project structure design allows easier re-use, iteration and model switching.
* For mor thorough conclusions see markdown notes in `.\notebooks\ship_hw_EDA_experiments.ipynb`.

## Requirements

```txt
pandas
jupyter
matplotlib
seaborn
gdown
fastapi
uvicorn
scikit-learn
xgboost
fastapi
uvicorn
python >= 3.10
```
## Author

**Pēteris**
