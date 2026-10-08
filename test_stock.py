from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
from datetime import datetime, timedelta

db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

today = datetime.now().strftime('%Y-%m-%d')
three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

df = loader.load_asset(
    secid='SBER',
    asset_type='stock',
    from_date=three_years_ago,
    till_date=today,
    interval=24  # дневные свечи
)

if not df.empty:
    print("Данные по SBER (первые 5 строк):")
    print(df.head())
    print(f"Всего записей: {len(df)}")
else:
    print("Данные по SBER не загружены")