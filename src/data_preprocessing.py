import pandas as pd
import yaml

def preprocess(cfg):
    df = pd.read_csv(cfg['data']['raw'])
    df.fillna(df.median(numeric_only=True), inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    df.to_csv(cfg['data']['processed_path'], index=False)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("params.yml"))
    preprocess(cfg)
