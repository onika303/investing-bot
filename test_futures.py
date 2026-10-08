from src.database import DatabaseManager
from src.moex_loader import MOEXLoader

db = DatabaseManager('finance.db')
loader = MOEXLoader(db)

# Загружаем непрерывный ряд для Si с дневными свечами
df = loader.load_asset(
    secid='Si',
    asset_type='futures',
    from_date='2023-01-01',
    till_date='2025-02-10',
    interval=24  # дневные свечи
)

if not df.empty:
    print("Успешно загружены данные по Si:")
    print(df.head())
    print(f"Всего записей: {len(df)}")
else:
    print("Не удалось загрузить данные для Si")