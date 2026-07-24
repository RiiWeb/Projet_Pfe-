import sqlite3
from pathlib import Path

db = Path('fraud.db')
if not db.exists():
    print('Database not found')
    raise SystemExit(1)

conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute('SELECT * FROM feedback ORDER BY created_at DESC, id DESC LIMIT 5')
rows = cur.fetchall()
if not rows:
    print('No feedback entries found')
else:
    for r in rows:
        print('---')
        for k in r.keys():
            print(f"{k}: {r[k]}")
conn.close()
