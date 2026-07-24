import sqlite3
from pathlib import Path
p = Path('instance') / 'fraud.db'
if not p.exists():
    print('Database not found at', p)
    raise SystemExit(1)
conn = sqlite3.connect(str(p))
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute('SELECT id, full_name, email, subject, category, message, created_at FROM feedback ORDER BY created_at DESC, id DESC LIMIT 5')
rows = cur.fetchall()
if not rows:
    print('No feedback entries found')
else:
    for r in rows:
        print('---')
        for k in r.keys():
            print(f"{k}: {r[k]}")
conn.close()
