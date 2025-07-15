import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib, yaml, mlflow, mlflow.sklearn

def train(cfg):
    df = pd.read_csv(cfg["data"]["processed"])
    X, y = df.drop("SalePrice", axis=1), df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["model"]["test_size"], random_state=cfg["model"]["random_state"]
    )

    mlflow.set_experiment("HousePrice")
    with mlflow.start_run():
        mlflow.log_params({k: cfg["model"][k] for k in ["n_estimators", "max_depth"]})

        model = RandomForestRegressor(**cfg["model"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        joblib.dump(model, "model.joblib")
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("params.yml"))
    train(cfg)
