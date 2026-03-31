import pandas as pd
import joblib
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import zipfile

# LOAD DATA

def load_data(path):
    
    if path.endswith(".zip"):
        with zipfile.ZipFile(path, 'r') as z:
            file_list = z.namelist()
            
            csv_file = [f for f in file_list if f.endswith(".csv")][0]
            
            with z.open(csv_file) as f:
                df = pd.read_csv(f)
    
    else:
        df = pd.read_csv(path)

    print("Dataset loaded successfully")
    print("Data Shape:", df.shape)

    return df


# SPLIT FEATURES & TARGET

def split_features_target(df, target="audience_count"):
    drop_cols = ["show_date", "book_theater_id"]

    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[target])
    y = df[target]

    print(" Features & Target separated")
    print("X shape:", X.shape, "| y shape:", y.shape)

    return X, y


# MODEL EVALUATION

def evaluate_model(y_true, y_pred):

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n Model Performance:")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    return rmse, mae, r2


# SAVE MODEL

def save_model(model, filename="best_model.pkl"):
    joblib.dump(model, filename)
    print(f"\n Model saved successfully as {filename}")


# LOAD MODEL

def load_model(filename="best_model.pkl"):
    model = joblib.load(filename)
    print(" Model loaded successfully")
    return model