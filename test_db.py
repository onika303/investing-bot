from src.database import DatabaseManager

db = DatabaseManager('finance.db')
# Проверяем создание таблиц
print("Таблицы созданы")

# Вставляем тестовые данные
import pandas as pd
import time

df = pd.DataFrame([{
    'secid': 'SBER',
    'ts': int(time.time()),
    'open': 280.0,
    'high': 282.0,
    'low': 279.0,
    'close': 281.0,
    'volume': 1000,
    'oi': 0
}])
db.insert_raw_candles(df)

# Проверяем чтение
last = db.get_last_ts('SBER')
print(f"Последний ts для SBER: {last}")

# Проверяем мета
db.upsert_meta('SBER', last, 'stock', 1)
meta = db.fetch_one('SELECT * FROM meta WHERE secid = ?', ('SBER',))
print(meta)