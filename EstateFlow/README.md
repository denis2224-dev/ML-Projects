# 🏠 EstateFlow - Finding Undervalued Real Estate with ML (Regression + kNN)

EstateFlow is an end-to-end machine learning pipeline that identifies **potentially undervalued properties** by combining:

- a **global pricing model** (Multiple Linear Regression), and  
- a **local market comparison** (k-Nearest Neighbors).

A property is flagged only when **both** the regression model and its local neighborhood comparison indicate it is priced below expectation - a simple but realistic “two-signal” approach similar to how analysts sanity-check valuations.

---

## ❓ What Problem Does EstateFlow Solve?

Real estate prices depend on both **global market trends** and **local neighborhood dynamics**.  
Using a single model often leads to false signals: global models miss local effects, while local comparisons lack broader context.

**EstateFlow solves this by combining both views.**

It identifies properties that are priced below:
- a **global regression-based estimate**, and
- the **average price of similar nearby properties (kNN)**.

Only listings that are undervalued by **both** measures are flagged as **Strong Buy** candidates.

This reduces false positives and produces rankings that align with real-world valuation logic.


---

## ✨ Key Features

- **SQL as source of truth** (SQLite): data is loaded from a CSV into a `listings` table.
- **Regression model (global signal)**: predicts `price_per_unit` from numeric drivers.
- **kNN similarity engine (local signal)**: compares each listing to its nearest neighbors by location + age.
- **Undervaluation scoring + ranking**: produces ranked listings and a filtered “Strong Buy” list.
- **Dockerized**: reproducible runs on any machine (recruiter-friendly).

---

## 📦 Dataset

This project uses the well-known *Real estate valuation* dataset (often shared on Kaggle/UCI).  
Columns (after normalization) include:

- `x2_house_age`
- `x3_distance_to_the_nearest_mrt_station`
- `x4_number_of_convenience_stores`
- `x5_latitude`
- `x6_longitude`
- `y_house_price_of_unit_area` (target)

---

## 🧠 Method

### 1) Global pricing model (Regression)
We fit a regression pipeline to estimate expected price using:

**Features**
- `x2_house_age`
- `x3_distance_to_the_nearest_mrt_station`
- `x4_number_of_convenience_stores`

**Target**
- `y_house_price_of_unit_area`

Pipeline:
- median imputation (robust to missing values)
- standard scaling
- linear regression

We also compute cross-validation metrics to estimate generalization stability.

---

### 2) Local market comparison (kNN neighbors)
For each listing, we find `k` nearest neighbors using:

**kNN Features**
- `x5_latitude`
- `x6_longitude`
- `x2_house_age`

We compute:
- neighbor IDs
- **mean neighbor price**

This gives a “local fair price” reference.

---

### 3) Undervaluation logic (“Strong Buy”)
We compute two gaps:

- **Regression gap**  

$$
\text{gap}_{\text{reg}} = \frac{\text{predicted} - \text{actual}}{\text{predicted}}
$$

- **Neighbor gap**  
$$
\text{gap}_{\text{knn}} = \frac{\text{neighbor\mean} - \text{actual}}{\text{neighbor\mean}}
$$


### ✅ Decision rule

A property is flagged as **Strong Buy** if **both** conditions hold:

$$
\text{gap}_{\text{reg}} \ge 0.10
\quad \text{and} \quad
\text{gap}_{\text{knn}} \ge 0.10
$$

This means the listing is priced **at least 10% below**:
- the global regression estimate, and
- the local neighborhood average.

### 💡 Interpretation

- **Regression gap** answers: *“Is this property cheap compared to the overall market model?”*
- **Neighbor gap** answers: *“Is this property cheap compared to similar nearby properties?”*

Only listings that pass **both checks** are considered strong opportunities.

---

## 📁 Project Structure

```text
EstateFlow/
│
├── data/
│   └── real_estate_data.csv
│
├── scripts/
│   ├── _dev_tools          # helpers
│   ├── db_setup.py         # CSV -> SQLite (creates table: listings)
│   ├── data_access.py      # SQL -> pandas loader
│   ├── train_model.py      # regression pipeline + CV + holdout metrics
│   ├── knn_engine.py       # neighbor search + neighbor mean price
│   └── evaluate.py         # merges signals, ranks, exports outputs
│
├── outputs/                # generated results (csv)
│   ├── ranked_listings.csv
│   └── strong_buys.csv
│
├── database.sqlite         # generated DB (optional to keep locally)
├── requirements.txt
└── Dockerfile
```

---

## 📝 Notes / Engineering Choices

This project intentionally prioritizes **clarity, reproducibility, and correctness** over unnecessary complexity.  
Key design decisions are outlined below.

### 🔹 SQL as the Source of Truth
Instead of training directly from a CSV, the dataset is first loaded into a **SQLite database**.
- Mirrors real-world ML workflows where data lives in databases or warehouses
- Decouples data ingestion from modeling logic
- Makes the pipeline easy to extend to PostgreSQL or cloud databases

---

### 🔹 Pipeline-Based Modeling
All preprocessing and modeling steps are wrapped in **scikit-learn Pipelines**.
- Prevents data leakage between train and test sets
- Keeps preprocessing consistent across training, validation, and inference
- Enables easy extension to cross-validation and grid search

---

### 🔹 Regression + kNN as Complementary Signals
The project deliberately avoids relying on a single model:
- **Regression** captures global market trends
- **kNN** captures local neighborhood effects

A property is flagged only when **both models agree**, reducing false positives and improving interpretability.

---

### 🔹 kNN Used for Similarity, Not Prediction
kNN is implemented using `NearestNeighbors`, not `KNeighborsRegressor`.
- The goal is **neighborhood comparison**, not prediction
- This makes the local price signal more transparent and easier to explain

---

### 🔹 Cross-Validation for Stability Checks
In addition to a train/test split, **k-fold cross-validation** is used to estimate generalization stability.
- Helps detect overly optimistic single-split results
- Guides model selection without over-tuning

---

### 🔹 Reproducibility via Docker
The entire pipeline is containerized using Docker.
- Ensures identical execution across machines
- Eliminates environment and dependency issues
- Allows the project to be run with a single command

---

### 🔹 Minimal Dependencies
The project intentionally uses a **minimal dependency set**:
- `numpy`
- `pandas`
- `scikit-learn`

This keeps Docker builds fast, avoids version conflicts, and improves portability.

---

### 🔹 Explicit Over Implicit
The code favors:
- clear variable names
- explicit intermediate columns (e.g. gaps, scores)
- readable decision rules

This makes the system easier to debug, audit, and explain to non-ML stakeholders.

---

### 🔹 Designed for Extension
The current architecture supports future upgrades such as:
- regularized regression (Ridge/Lasso)
- configurable CLI thresholds
- persisted trained models
- alternative distance metrics
- production databases (PostgreSQL)

Without requiring major refactoring.
