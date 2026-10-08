from src.database import DatabaseManager
db = DatabaseManager('finance.db')
# Посмотреть список уникальных secid в raw_candles
rows = db.fetch_all("SELECT DISTINCT secid FROM raw_candles ORDER BY secid")
print("Загруженные активы:", [r[0] for r in rows])
# Посмотреть количество записей по каждому
for secid, in rows:
    cnt = db.fetch_one("SELECT COUNT(*) FROM raw_candles WHERE secid = ?", (secid,))
    print(f"{secid}: {cnt[0]} записей")