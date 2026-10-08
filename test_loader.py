from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
import asyncio

db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

# Загружаем данные по Сберу за последние 5 дней
df = loader.fetch_candles(
    secid='SBER',
    asset_type='stock',
    from_date='2025-02-01',
    till_date='2025-02-10',
    interval=1
)
print(df.head())
print(f"Загружено {len(df)} свечей")

# Сохраняем в БД
if not df.empty:
    db.insert_raw_candles(df)
    last_ts = df['ts'].max()
    db.upsert_meta('SBER', last_ts, 'stock', len(df))