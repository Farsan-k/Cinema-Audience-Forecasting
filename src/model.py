import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from utils import load_data, split_features_target, evaluate_model, save_model
from preprocessing import preprocessing_pipeline

df = load_data("train.zip")
X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "LightGBM": LGBMRegressor(),
    "XGBoost": XGBRegressor(),
    "GradientBoosting": GradientBoostingRegressor()
}

best_score = -1
best_pipe = None
best_name = None
best_rmse = None
best_mae = None

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipe = Pipeline([
        ("preprocessor", preprocessing_pipeline),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse, mae, r2 = evaluate_model(y_test, y_pred)

    if r2 > best_score:
        best_score = r2
        best_pipe = pipe
        best_name = name
        best_rmse = rmse
        best_mae = mae

print(f"\nBest Model: {best_name}")
print(f"Baseline RMSE: {best_rmse}")
print(f"Baseline MAE: {best_mae}")
print(f"Baseline R2: {best_score}")

best_base_model = best_pipe.named_steps["model"]

tscv = TimeSeriesSplit(n_splits=5)

if best_name == "LightGBM":
    param_random = {
        "model__num_leaves": [31, 50, 100],
        "model__max_depth": [-1, 5, 10, 20],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__n_estimators": [100, 200, 500]
    }
elif best_name == "RandomForest":
    param_random = {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10]
    }
elif best_name == "XGBoost":
    param_random = {
        "model__n_estimators": [100, 200, 500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, 7]
    }
elif best_name == "GradientBoosting":
    param_random = {
        "model__n_estimators": [100, 200, 500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, 7]
    }
elif best_name == "DecisionTree":
    param_random = {
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10]
    }
else:
    param_random = {}

tuning_pipeline = Pipeline([
    ("preprocessor", preprocessing_pipeline),
    ("model", best_base_model)
])

random_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_random,
    n_iter=10,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("\nTuning Best CV Score:", random_search.best_score_)

y_pred = best_model.predict(X_test)

rmse, mae, r2 = evaluate_model(y_test, y_pred)

save_model(best_model, "best_model.pkl")



# grid_search = GridSearchCV(
#     estimator=tuning_pipeline,
#     param_grid=param_random,
#     cv=tscv,
#     scoring="neg_root_mean_squared_error",
#     n_jobs=1
# )

# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_

# print("\nTuning Best CV Score:", grid_search.best_score_)

# y_pred = best_model.predict(X_test)

# rmse, mae, r2 = evaluate_model(y_test, y_pred)

# save_model(best_model, "best_model.pkl")

