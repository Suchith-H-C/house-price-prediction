import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib, yaml, mlflow
import mlflow.sklearn

def train(cfg):
    # Load processed data
    df = pd.read_csv(cfg["data"]["processed_path"])
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"]
    )

    mlflow.set_experiment("HousePricePrediction")
    with mlflow.start_run(run_name="Training"):
        # Log model parameters
        for param in cfg["model"]:
            mlflow.log_param(param, cfg["model"][param])

        # Train model
        model = RandomForestRegressor(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            random_state=cfg["model"]["random_state"]
        )
        model.fit(X_train, y_train)

        # Predictions and metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("train_mse", mse)
        mlflow.log_metric("train_rmse", rmse)
        mlflow.log_metric("train_mae", mae)
        mlflow.log_metric("train_r2_score", r2)

        # Save model
        joblib.dump(model, "model.joblib")
        mlflow.sklearn.log_model(model, "model")

        # ✅ Save expected columns
        joblib.dump(X.columns.tolist(), "columns.pkl")

        print(f"✅ Model trained and logged. RMSE: {rmse:.2f}, R2: {r2:.4f}")

if __name__ == "__main__":
    with open("params.yml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
