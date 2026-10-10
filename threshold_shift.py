"""Evaluate a chosen threshold. Choose it on validation, never on test results."""
import argparse,json
import pandas as pd
from evaluation import metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument("csv",nargs="?",default="artifacts/validation_preds.csv");ap.add_argument("--threshold",type=float,default=.4)
    args=ap.parse_args();df=pd.read_csv(args.csv)
    if not df.Actual.isin(["Malignant","Benign"]).all():raise ValueError("Unexpected class label")
    print(json.dumps(metrics((df.Actual=="Malignant").astype(int),df["Prediction Value"],args.threshold),indent=2))
if __name__=="__main__":main()
