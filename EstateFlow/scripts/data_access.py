import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "database.sqlite"

def load_listings():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM listings", conn)
    conn.close()
    return df

if __name__ == "__main__":
    df = load_listings()
    print(df.head())
    print(f"Rows loaded: {len(df)}")