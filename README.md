# Ship Propulsion Prediction
***
## Task
Implement machine learning pipeline and deploy it via API to predict vessel **shaftPower** and **speedOverGround** from environmental and engine data, including full support for:
- Data preprocessing
- Feature engineering
- Model training, evaluation, and hyperparameter tuning
- Model deployment via API
The goal is to select the best feature set, model, and parameters—and serve predictions through a robust FastAPI app.

## Description
### 1. Model Development
#### 1.1. Predicts two target variables:
* `shaftPower`
* `speedOverGround`

#### 1.2. Input features from Ship Power Prediction Dataset:
* A):
    * : All 37 features, fully cleaned and engineered.
* B):
    * selecting and engineering narrower set of Task-specific 10 features:
        * **Weather data**: `windSpeed`, `windDirection`, `waveHeight`, `waveDirection`, `swellHeight`, `swellDirection`, `currentSpeed`, `currentDirection`, `airTemperature`
        * **Engine data**: `mainEngineMassFlowRate`

#### 1.3. Data Preprocessing

* Load `.json` dataset into a pandas `DataFrame`
* Handle missing values (e.g., replace NaNs with 0s or drop rows)
* Normalize features using `StandardScaler`
* Drop irrelevant or highly collinear features
* Optional: use `--task_features` flag to train only on features mentioned in the task
* Use the `FeatureSelector` class to dynamically remove features based on task flags or correlation thresholds.

#### 1.4. Feature Engineering

Implemented via a custom `FeatureEngineer` class:
* For case A) - all features utilized:
    * Angular decomposition of direction-based features (e.g., wind/heading)
    * Force vector breakdowns for environment force features
    * Time-based and derived features
    * adjust time and directional features for preserving circularity (cos/sin)
* For case B) - narrower, task specific, features set used:
    * adjust directional features for preserving circularity (cos/sin)

#### 1.5. Model Selection
- See model comparison in `.\notebooks\ship_hw_EDA_experiments.ipynb`.

- Final setup used:
    * `RegressorChain` wrapper with XGBoost (`XGBRegressor`)
    * Optional tuning with:
        * `RandomizedSearchCV` + `GridSearchCV`
        * Auto-sampled hyperparameter ranges for`GridSearchCV` from normal distributions around `RandomizedSearchCV` best results.

#### 1.6. Evaluation Metrics
* R² Score
* RMSE
* MAE
* MAPE (masked for 0s)
* Highest raw data Feature–target correlation score for baseline

### 2: FastAPI Deployment
API located in api/. Serves ML model predictions.
**Important** Before running the API, make sure trained models and preprocessors are present. Use train_model.py or download from external source.
Test api with the same input features as you trained your model with.

#### App Key components:
- Contains home `/` endpoint with welcome message.
- Contains the `/predict` endpoint which:
    - Accepts JSON input with environmental, engine, and vessel data.
    - Selects the appropriate model dynamically from `models/` directory based on feature count provided as input. Two options are available - task specific limited model for 10 feature input of full feature set model for 37 feature input.
- Pydantic models for request validation (PropulsionInputFull) and response formatting (PropulsionOutput).
- Returns predicted propulsion metrics as a structured JSON response.
- Models and preprocessors loaded at startup.
- Command-line script to test the API using real samples from test datasets (json test file created in `/data` folder during train-test-split with `/scripts/data/split.py`).

#### Example API request:
```
JSON
{
      "windSpeed": 6.699999809265137,
      "windDirection": 161.8000030517578,
      "waveHeight": 1.2000000476837158,
      "waveDirection": 79.30000305175781,
      "swellHeight": 0.7900000214576721,
      "swellDirection": 33.810001373291016,
      "currentSpeed": 0.10000000149011612,
      "currentDirection": 343.6000061035156,
      "airTemperature": 5.599999904632568,
      "mainEngineMassFlowRate": 196.83099365234375
    }
```

#### Example response:
```
JSON
{
    "shaftPower": 892461.25,
    "speedOverGround": 10.47188663482666
}
```

## Project Structure
```
.
├── .env                         # Project root Python path reference
├── .gitignore                   # Git ignore file
├── config.yaml                  # Project configuration file
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── train_model.py               # Entry point (train, evaluate, tune)
│
├── api                          # FastAPI application
│   ├── app.py                   # Main FastAPI app
│   ├── schema.py                # Pydantic model for input/output validation
│   ├── test_api.py              # API test script
│   │
│   ├── model
│   │   └── load_models.py       # Load trained models
│   │
│   ├── routers
│   │   ├── home.py              # Base/root route
│   │   └── predictions.py       # Prediction endpoint
│   │
│   └── utils
│       ├── handlers.py          # Request handlers
│       └── preprocessing.py     # Preprocessing utilities
│
├── data                         # Input / test data
│   ├── api_test_feature_target_pairs.json
│   ├── api_test_feature_target_pairs_short.json
│   └── test-assignment-dataset.json
│
├── models                       # Trained models and preprocessors
│   ├── final_model.pkl
│   ├── tuning_prep.pkl
│   └── full_feature_set
│       ├── final_model.pkl
│       └── tuning_prep.pkl
│
├── notebooks                    # Jupyter experimentation notebooks
│   └── ship_hw_EDA_experiments.ipynb
│
└── scripts                      # Data and model scripts
    ├── config.py                # Script-level config
    │
    ├── data                     # Data pipeline scripts (cleaning, loading, feature prep)
    │   ├── clean.py
    │   ├── feature_engineer.py
    │   ├── feature_select.py
    │   ├── load.py
    │   └── split.py
    │
    ├── model                    # Model training, tuning, evaluation
    │   ├── evaluation.py
    │   ├── get_model.py
    │   └── tuning.py
    │
    └── utils                    # EDA tools (e.g. correlation heatmap)
        ├── eda.py
        └── visual.py
```

## Usage
#### 1. Install Dependencies
Clone the [git repository](https://github.com/ButrosV/ship_prop_model.git) or download `.\notebooks\ship_hw_EDA_experiments.ipynb`.

```bash
pip install -r requirements.txt
```
If .env isn't picked up by your editor:
Set PYTHONPATH manually from project root:
```export PYTHONPATH=$(pwd)```       # macOS/Linux
```$env:PYTHONPATH = (Get-Location)```  # PowerShell

#### 2.1. Train Models
Train and evaluate the model using default model and benchmark hyperparameters from config file:

```bash
python train_model.py
```

Train with full hyperparameter tuning (~15 hrs), slightly improves results over benchmark hyperparameters:

```bash
python train_model.py --tune_model
```

Train using only task-specified features:
```bash
python train_model.py --task_features
```

##### 2.2. Alternatively run jupyter notebook:
**WARNING** Running entire notebook can take up to 20h and requires ~ 5GB of space for saving model selection iterations.

Run all cells in `.\notebooks\ship_hw_EDA_experiments.ipynb` file to:
- Follow exploratory data analysis;
- Follow model selection (by default saves all experimental models, saving can be turned off manually withinin the notebook);
- tune hyperparameters for selected model;
- save final model for later use with FastAPI.

#### 3. Run FastAPI App
```bash
uvicorn api.app:app --reload
```
Open docs for additional endpoint documentation and testing: http://127.0.0.1:8000/docs

### 4. Test API
While application is running:
 (task feature model trained on limited set of features, test input taken from JSON file created during train-test-split operation before training models):
```bash
python -m api.test_api
```

To test application with model trained on full set of features:
```bash
python -m api.test_api --all_features
```

To provide your own test file (must be JSON, adhering to formatting requirements: top level 'features' and 'targets' keys):
```bash
python -m api.test_api --source_file data/custom_test_input.json
```

### Clean up your working directory
- To remove data amn models folder contents:
```bash
rm -rf data/* models/*
```
```powershell
Remove-Item -Recurse -Force .\data\* , .\models\*
```
- To clean entire current root directory:
```bash
rm -rf ./*
```
```powershell
Remove-Item -Recurse -Force .\*
```

## Configuration

YAML config file includes:
* Paths to model and data directories 
* Default preprocessor and model file names
* Default model parameters
* Feature drop lists
* Default column name definitions for targets and feature engineering defaults

## Design Justification

* **RegressorChain** allows leveraging multi-output dependencies (shaftPower ↔ speedOverGround).
* **XGBoost** provides:
    * Acceptable evaluation metric performance.
    * Less costly in terms of computational resources (training times and RAM requirements) when compared to `RandomForest`.
    * Thus more suitable for hyperparameter tuning.
* Custom feature engineering enables richer directional awareness, interactions among forces and potential seasonal weather patterns.
* Modular project structure design allows easier re-use, iteration and model switching.
* See EDA and experimentation notebook for full model selection rationale `.\notebooks\ship_hw_EDA_experiments.ipynb`.

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

python >= 3.10
```
## Author

**Pēteris**
