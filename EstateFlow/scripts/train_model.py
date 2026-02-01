from data_access import load_listings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


FEATURES = [
    "x2_house_age",
    "x3_distance_to_the_nearest_mrt_station",
    "x4_number_of_convenience_stores",
]
TARGET = "y_house_price_of_unit_area"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def train_regression(test_size: float = 0.2, random_state: int = 42, cv: int = 5) -> Pipeline:
    df = load_listings()

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DB: {missing}\nAvailable: {list(df.columns)}")

    X = df[FEATURES]
    y = df[TARGET]

    pipeline = build_pipeline()

    cv_r2 = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    cv_mse = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error")
    cv_rmse = cv_mse ** 0.5

    print(f"Cross-Validation (cv={cv})")
    print(f"R²   : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    print(f"RMSE : {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("Holdout Results (Train/Test Split)")
    print(f"n_train: {len(X_train)} | n_test: {len(X_test)}")
    print(f"R²  : {r2:.3f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    model = pipeline.named_steps["model"]
    print("\nCoefficients (standardized)")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"intercept: {model.intercept_:.4f}")

    return pipeline


if __name__ == "__main__":
    train_regression()