import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data_access import load_listings


KNN_FEATURES = ["x5_latitude", "x6_longitude", "x2_house_age"]
PRICE_COL = "y_house_price_of_unit_area"
ID_COL = "no"


def compute_knn_neighbors(k: int = 5) -> pd.DataFrame:

    df = load_listings()

    required = KNN_FEATURES + [PRICE_COL, ID_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DB: {missing}\nAvailable: {list(df.columns)}")

    X = df[KNN_FEATURES]
    y_price = df[PRICE_COL].to_numpy()
    ids = df[ID_COL].to_numpy()

    prep = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_scaled = prep.fit_transform(X)

    # +1 - the nearest neighbor of each point is itself (distance 0)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_scaled)

    distances, indices = nn.kneighbors(X_scaled, return_distance=True)

    # exclude itself at position 0
    neighbor_indices = indices[:, 1:]  # shape: (n_samples, k)

    neighbor_mean_price = np.array([y_price[idxs].mean() for idxs in neighbor_indices])
    neighbor_id_lists = [ids[idxs].tolist() for idxs in neighbor_indices]

    result = pd.DataFrame(
        {
            ID_COL: ids,
            "actual_price": y_price,
            "knn_neighbor_mean_price": neighbor_mean_price,
            "knn_neighbor_ids": neighbor_id_lists,
        }
    )

    return result


if __name__ == "__main__":
    out = compute_knn_neighbors(k=5)
    print(out.head())
    print("\nSummary:")
    print(out[["actual_price", "knn_neighbor_mean_price"]].describe())