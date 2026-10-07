import sqlite3
conn = sqlite3.connect('ml_detective_dev.db')
cursor = conn.execute('SELECT id, dataset_name, health_score, grade, created_at FROM investigations')
for row in cursor.fetchall():
    print(row)
conn.close()