import pandas as pd
import joblib
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

def evaluate(cfg):
    # Load data and model
    df = pd.read_csv(cfg["data"]["processed_path"])
    model = joblib.load("model.joblib")

    # Split the dataset
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"]
    )

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log to MLflow
    mlflow.set_experiment("HousePricePrediction")
    with mlflow.start_run(run_name="ModelEvaluation"):
        mlflow.log_metric("eval_rmse", rmse)
        mlflow.log_metric("eval_mae", mae)
        mlflow.log_metric("eval_r2_score", r2)

    # Print summary
    print(f"✅ Evaluation complete — RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

if __name__ == "__main__":
    with open("params.yml") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg)
