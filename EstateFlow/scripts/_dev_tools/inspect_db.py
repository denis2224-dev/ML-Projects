import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "database.sqlite"

def inspect():
    print(f"DB path: {DB_PATH}")
    print(f"DB exists: {DB_PATH.exists()}  |  size: {DB_PATH.stat().st_size if DB_PATH.exists() else 0} bytes")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # list all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    print("Tables:", tables)

    # if listings exists, count rows
    if "listings" in tables:
        cur.execute("SELECT COUNT(*) FROM listings;")
        print("listings row count:", cur.fetchone()[0])
    else:
        print("No 'listings' table found.")

    conn.close()

if __name__ == "__main__":
    inspect()