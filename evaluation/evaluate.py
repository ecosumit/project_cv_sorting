from __future__ import annotations
import argparse, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

def main(pred_csv: str, truth_csv: str, label_col: str = "label"):
    pred = pd.read_csv(pred_csv)
    truth = pd.read_csv(truth_csv)
    df = pred.merge(truth, on="file", how="inner", suffixes=("_pred",""))
    y_true = (df[label_col].astype(int)).values
    # assume we threshold total_score at 0.5
    y_pred = (df["total_score"] >= 0.5).astype(int).values
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    ap = average_precision_score(y_true, df["total_score"].values)
    print(f"Precision: {p:.3f}\nRecall: {r:.3f}\nF1: {f1:.3f}\nAverage Precision (AP): {ap:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--truth_csv", required=True)
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()
    main(args.pred_csv, args.truth_csv, args.label_col)
