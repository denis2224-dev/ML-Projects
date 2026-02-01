# scripts/evaluate.py

from pathlib import Path
import pandas as pd
import argparse

from data_access import load_listings
from train_model import train_regression, FEATURES as REG_FEATURES
from knn_engine import compute_knn_neighbors


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# thresholds (tune later)
REG_UNDERVALUE_THRESHOLD = 0.10   # 10% under regression prediction
KNN_UNDERVALUE_THRESHOLD = 0.10   # 10% under neighbor mean
TOP_N = 20


def evaluate_undervalued(k: int = 5) -> pd.DataFrame:
    df = load_listings().copy()

    pipeline = train_regression()
    df["regression_pred"] = pipeline.predict(df[REG_FEATURES])

    knn_df = compute_knn_neighbors(k=k)

    merged = df.merge(knn_df, on="no", how="inner")

    merged["regression_gap"] = (merged["regression_pred"] - merged["actual_price"]) / merged["regression_pred"].replace(0, pd.NA)
    merged["neighbor_gap"] = (merged["knn_neighbor_mean_price"] - merged["actual_price"]) / merged["knn_neighbor_mean_price"].replace(0, pd.NA)

    merged["strong_buy"] = (
        (merged["regression_gap"] >= REG_UNDERVALUE_THRESHOLD)
        & (merged["neighbor_gap"] >= KNN_UNDERVALUE_THRESHOLD)
    )

    merged["undervalue_score"] = merged["regression_gap"].fillna(0) + merged["neighbor_gap"].fillna(0)

    low, high = merged["actual_price"].quantile([0.01, 0.99])
    merged = merged[(merged["actual_price"] >= low) & (merged["actual_price"] <= high)]

    ranked = merged.sort_values("undervalue_score", ascending=False)

    cols = [
        "no",
        "actual_price",
        "regression_pred",
        "knn_neighbor_mean_price",
        "regression_gap",
        "neighbor_gap",
        "undervalue_score",
        "strong_buy",
        "knn_neighbor_ids",
        "x2_house_age",
        "x3_distance_to_the_nearest_mrt_station",
        "x4_number_of_convenience_stores",
        "x5_latitude",
        "x6_longitude",
    ]

    return ranked[cols]


def main():
    ranked = evaluate_undervalued(k=5)

    out_all = OUTPUT_DIR / "ranked_listings.csv"
    out_strong = OUTPUT_DIR / "strong_buys.csv"

    ranked.to_csv(out_all, index=False)
    ranked[ranked["strong_buy"]].head(TOP_N).to_csv(out_strong, index=False)

    print(f"Saved: {out_all}")
    print(f"Saved: {out_strong}")

    print("\nTop Strong Buys:")
    print(ranked[ranked["strong_buy"]].head(TOP_N)[
        ["no", "actual_price", "regression_pred", "knn_neighbor_mean_price", "undervalue_score"]
    ])

    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--k", type=int, default=5)
        p.add_argument("--reg_thr", type=float, default=0.10)
        p.add_argument("--knn_thr", type=float, default=0.10)
        p.add_argument("--top_n", type=int, default=20)
        return p.parse_args()


if __name__ == "__main__":
    main()