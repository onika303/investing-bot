from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
from datetime import datetime, timedelta

db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

today = datetime.now().strftime('%Y-%m-%d')
three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

# Загружаем данные по контракту SiU6 (или любому другому из списка)
df = loader.fetch_candles(
    secid='SiU6',  # замените на существующий контракт, если SiU6 не активен
    asset_type='futures',
    from_date=three_years_ago,
    till_date=today,
    interval=24  # дневные свечи
)

print(df.head(10))
print(f"Всего записей: {len(df)}")