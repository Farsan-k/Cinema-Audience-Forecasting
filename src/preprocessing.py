import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder



class ColumnDropper(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')


# 1. SIMPLE LABEL ENCODER

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, column):
        self.column = column
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        if self.column in X.columns:
            self.encoder.fit(X[self.column].astype(str))
        return self

    def transform(self, X):
        X = X.copy()
        
        if self.column in X.columns:
            X[self.column] = self.encoder.transform(X[self.column].astype(str))
        
        return X


# 2. LAG FEATURE IMPUTER

class LagFeatureImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lag_columns):
        self.lag_columns = lag_columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.copy()
        
        for col in self.lag_columns:
            if col not in x.columns:
                continue

            lag_n = int(col.split("_")[1])
            values = x[col].values.copy()
            
            i = 0
            while i < len(values):
                
                if pd.isna(values[i]):
                    start = i
                    while i < len(values) and pd.isna(values[i]):
                        i += 1
                    end = i
                    
                    nan_length = end - start
                    
                    next_vals = values[end:end+lag_n]
                    next_vals = next_vals[~pd.isna(next_vals)]
                    
                    if len(next_vals) >= nan_length:
                        for j in range(nan_length):
                            values[start + j] = next_vals[j]
                    else:
                        prev_vals = values[max(0, start-lag_n):start]
                        prev_vals = prev_vals[~pd.isna(prev_vals)]
                        
                        if len(prev_vals) > 0:
                            fill_val = np.mean(prev_vals)
                            values[start:end] = fill_val
                
                else:
                    i += 1
            
            x[col] = values
        
        return x


# 3. ROLLING MEAN IMPUTER

class RollingFeatureImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, rolling_columns):
        self.rolling_columns = rolling_columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.copy()
        
        for col in self.rolling_columns:
            if col in x.columns:
                x[col] = x[col].ffill().bfill()
        
        return x


# 4. DERIVED FEATURE BUILDER

class DerivedFeatureBuilder(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.copy()
        
        required_cols = ["lag_1", "lag_3", "lag_7", "lag_14"]
        if not all(col in x.columns for col in required_cols):
            return x
        
        x["trend_7"] = x["lag_1"] - x["lag_7"]
        x["trend_14"] = x["lag_1"] - x["lag_14"]

        x["diff_1_3"] = x["lag_1"] - x["lag_3"]
        x["diff_7_14"] = x["lag_7"] - x["lag_14"]

        x["lag_ratio_7"] = x["lag_1"] / (x["lag_7"] + 1)
        
        return x


# 5. ROLLING STD CREATOR

class RollingStdFeatureCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self, group_col, windows=(7, 14)):
        self.group_col = group_col
        self.windows = windows

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.copy()

        if self.group_col not in x.columns:
            return x

        for window in self.windows:
            col_name = f"roll_std_{window}"
            
            x[col_name] = x.groupby(self.group_col)["lag_1"].transform(
                lambda s: s.rolling(window, min_periods=2).std()
            )

            x[col_name] = x[col_name].fillna(x[col_name].median())

        return x


# 6. FINAL CLEANER

def final_cleaning(x):
    x = x.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0)
    return x


# 7. COLUMN LISTS

lag_cols = ["lag_1", "lag_3", "lag_7", "lag_14"]
rolling_cols = ["roll_mean_7", "roll_mean_14"]


class DataFrameScaler(StandardScaler):
    
    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        X_scaled = super().transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns, index=X.index)


# 8. FULL PIPELINE

preprocessing_pipeline = Pipeline([
    
    ("label_encode", LabelEncoderTransformer(column="book_theater_id")),
    
    ("lag_imputer", LagFeatureImputer(lag_columns=lag_cols)),
    
    ("rolling_imputer", RollingFeatureImputer(rolling_columns=rolling_cols)),
    
    ("feature_builder", DerivedFeatureBuilder()),
    
    ("rolling_std", RollingStdFeatureCreator(
        group_col="book_theater_id"
    )),
    
    ("final_clean", FunctionTransformer(final_cleaning)),
    
    ("scaler", DataFrameScaler())
])