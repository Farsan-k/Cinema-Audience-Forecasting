import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from utils import load_data, split_features_target, evaluate_model, save_model
from preprocessing import preprocessing_pipeline, ColumnDropper

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
best_name = ""

results_list = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipe = Pipeline([
        ("drop_id", ColumnDropper(columns=[])),
        ("preprocessor", preprocessing_pipeline),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    rmse, mae, r2 = evaluate_model(y_test, y_pred)

    results_list.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    if r2 > best_score:
        best_score = r2
        best_pipe = pipe
        best_name = name

results_df = pd.DataFrame(results_list)

print("\nAll Model Results:")
print(results_df.sort_values(by="R2", ascending=False))

print(f"\nBest Model: {best_name} | R2 Score: {best_score:.4f}")

final_model = best_pipe.named_steps["model"]

X_processed = best_pipe[:-1].transform(X_train)

preprocessor = best_pipe.named_steps["preprocessor"]

try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = X_processed.columns

feature_names = np.array(feature_names)

if hasattr(final_model, "feature_importances_"):

    importance_score = final_model.feature_importances_

    min_len = min(len(feature_names), len(importance_score))
    feature_names = feature_names[:min_len]
    importance_score = importance_score[:min_len]

    indices = np.argsort(importance_score)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title(f"{best_name} Feature Importance")

    plt.bar(range(min_len), importance_score[indices])
    plt.xticks(range(min_len), feature_names[indices], rotation=90)

    plt.tight_layout()
    plt.show()

else:
    print(f"{best_name} does not support feature_importances_")

print("\nStarting LightGBM Hyperparameter Tuning...")

lgbm_pipeline = Pipeline([
    ("drop_id", ColumnDropper(columns=[])),
    ("preprocessor", preprocessing_pipeline),
    ("model", LGBMRegressor(random_state=42))
])

param_random = {
    "model__num_leaves": [31, 50, 100],
    "model__max_depth": [-1, 5, 10, 20],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__n_estimators": [100, 200, 500],
    "model__min_data_in_leaf": [10, 20, 50],
    "model__lambda_l1": [0.0, 0.1, 1.0],
    "model__lambda_l2": [0.0, 1.0, 5.0],
    "model__feature_fraction": [0.6, 0.8, 1.0],
    "model__bagging_fraction": [0.6, 0.8, 1.0],
    "model__bagging_freq": [0, 5],
    "model__max_bin": [255, 512],
}

tscv = TimeSeriesSplit(n_splits=5)

random_search = RandomizedSearchCV(
    estimator=lgbm_pipeline,
    param_distributions=param_random,
    n_iter=30,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest CV Score (RMSE):", random_search.best_score_)

print("\n===== Final LightGBM Performance =====")

y_pred = best_model.predict(X_test)

rmse, mae, r2 = evaluate_model(y_test, y_pred)

save_model(best_model, "best_model.pkl")

joblib.dump(X_train.columns.tolist(), "feature_order.pkl")

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

train_mean = -train_scores.mean(axis=1)
val_mean = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label="Train Error")
plt.plot(train_sizes, val_mean, label="Validation Error")
plt.legend()
plt.title("Learning Curve")
plt.show()

