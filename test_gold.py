from src.database import DatabaseManager
from src.moex_loader import MOEXLoader
from datetime import datetime, timedelta

db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

today = datetime.now().strftime('%Y-%m-%d')
three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

df = loader.load_asset(
    secid='GOLD',          # или 'GOLDM' для мини-фьючерса
    asset_type='futures',
    from_date=three_years_ago,
    till_date=today,
    interval=24            # дневные свечи
)

if not df.empty:
    print("Данные по GOLD (первые 5 строк):")
    print(df.head())
    print(f"Всего записей: {len(df)}")
    print(f"Диапазон дат: {df['ts'].min()} - {df['ts'].max()}")
else:
    print("Данные по GOLD не загружены")