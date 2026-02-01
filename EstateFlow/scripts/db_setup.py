import pandas as pd
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "real_estate_data.csv"
DB_PATH = PROJECT_ROOT / "database.sqlite"

def setup_database():
    if not CSV_PATH.exists():
        print(f"ERROR: Could not find {CSV_PATH}")
        return

    print(f"LOADING DATA FROM {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]

    print(f"CREATING SQL DATABASE: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    df.to_sql("listings", conn, if_exists="replace", index=False)

    count = pd.read_sql("SELECT COUNT(*) AS n FROM listings", conn)
    print(f"SUCCESS! Total rows in listings: {count.loc[0, 'n']}")

    conn.close()

if __name__ == "__main__":
    setup_database()